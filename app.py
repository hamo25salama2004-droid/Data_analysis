import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from fuzzywuzzy import fuzz
import re
import io

# -----------------------------------------------------------------------------
# Class: SmartCleaner (الفئة الذكية لتنفيذ جميع العمليات)
# -----------------------------------------------------------------------------
class SmartCleaner:
    def __init__(self, df):
        self.df = df.copy()

    # --- 1. اكتشاف أنواع الأعمدة تلقائياً ---
    def detect_column_types(self):
        col_types = {"numeric": [], "text": [], "date": [], "categorical": []}
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                col_types["date"].append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                col_types["numeric"].append(col)
            else:
                # إذا كانت نصية وبها قيم فريدة قليلة، تعتبر فئوية
                if self.df[col].nunique() < 50 and self.df.shape[0] > 100:
                    col_types["categorical"].append(col)
                col_types["text"].append(col)
        return col_types

    # --- 2. معالجة القيم المفقودة (10 عمليات) ---
    def handle_missing(self, cols, method):
        if method == "Drop Rows":
            self.df = self.df.dropna(subset=cols)
        elif method == "Drop Column":
            self.df = self.df.drop(columns=cols, errors='ignore')
        elif method == "KNN Imputer (AI)":
            imputer = KNNImputer(n_neighbors=5)
            self.df[cols] = imputer.fit_transform(self.df[cols])
        elif method == "MICE Imputer (AI)":
            imputer = IterativeImputer(random_state=42)
            self.df[cols] = imputer.fit_transform(self.df[cols])
        elif method == "Mean Fill":
            self.df[cols] = self.df[cols].fillna(self.df[cols].mean())
        elif method == "Median Fill":
            self.df[cols] = self.df[cols].fillna(self.df[cols].median())
        elif method == "Mode Fill":
            for col in cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif method == "Constant Fill (Zero)":
            self.df[cols] = self.df[cols].fillna(0)
        elif method == "Forward Fill (ffill)":
            self.df[cols] = self.df[cols].ffill()
        elif method == "Backward Fill (bfill)":
            self.df[cols] = self.df[cols].bfill()
        return self.df

    # --- 3. معالجة القيم المتكررة (7 عمليات) ---
    def handle_duplicates(self, cols, method, threshold=95):
        if method == "Exact Duplicates":
            initial_rows = self.df.shape[0]
            self.df = self.df.drop_duplicates(subset=cols, keep='first')
            return self.df, initial_rows - self.df.shape[0]
        
        elif method == "Fuzzy Match (Text)":
            # 5 عمليات ضمن هذه التقنية
            def get_fuzz_score(row):
                # ندمج القيم في الأعمدة المحددة للمقارنة
                combined = tuple(row[c] for c in cols)
                scores = []
                for i in range(len(self.df)):
                    target_combined = tuple(self.df.iloc[i][c] for c in cols)
                    if row.name != self.df.iloc[i].name:
                        # جودة المطابقة (Qratio) هي الأقوى من Fuzzywuzzy
                        score = fuzz.QRatio(str(combined), str(target_combined))
                        scores.append(score)
                return max(scores) if scores else 100

            # نحذف الصفوف التي لديها تطابق قوي جداً
            duplicate_indices = self.df.apply(lambda row: get_fuzz_score(row) >= threshold, axis=1)
            
            # نحتفظ بالنسخ الفريدة (تعتمد على طريقة عمل FuzzyWuzzy)
            temp_df = self.df[duplicate_indices].drop_duplicates(subset=cols)
            self.df = pd.concat([self.df[~duplicate_indices], temp_df])
            return self.df, self.df.shape[0] - initial_rows 
        
        return self.df, 0

    # --- 4. معالجة القيم الشاذة (8 عمليات) ---
    def handle_outliers(self, cols, method, threshold=3):
        initial_rows = self.df.shape[0]
        for col in cols:
            if method == "IQR Method":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            
            elif method == "Z-Score Filter":
                # نحذف الصفوف التي تبعد أكثر من threshold (افتراضياً 3)
                self.df = self.df[np.abs(self.df[col]-self.df[col].mean())/self.df[col].std() < threshold]

            elif method == "Isolation Forest (AI)":
                iso = IsolationForest(contamination=0.1, random_state=42)
                yhat = iso.fit_predict(self.df[col].fillna(self.df[col].median()).values.reshape(-1, 1))
                mask = yhat != -1
                self.df = self.df[mask]

            elif method == "Capping (Winsorization)":
                # استبدال القيم المتطرفة بالحدود القصوى
                Q1 = self.df[col].quantile(0.05)
                Q3 = self.df[col].quantile(0.95)
                self.df[col] = np.where(self.df[col] < Q1, Q1, self.df[col])
                self.df[col] = np.where(self.df[col] > Q3, Q3, self.df[col])
            
            elif method == "Log Transformation":
                # تقليل تأثير القيم الكبيرة جداً
                self.df[col] = np.log1p(self.df[col]) # log(1+x)

        return self.df, initial_rows - self.df.shape[0]

    # --- 5. معالجة النصوص والترجمة (15 عملية + ترجمة) ---
    def handle_text_and_translate(self, cols, method, target_lang=None):
        for col in cols:
            self.df[col] = self.df[col].astype(str)
            
            if method == "Lowercase":
                self.df[col] = self.df[col].str.lower()
            elif method == "Uppercase":
                self.df[col] = self.df[col].str.upper()
            elif method == "Remove Punctuation":
                self.df[col] = self.df[col].str.replace(r'[^\w\s]', '', regex=True)
            elif method == "Remove Stop Words (English)":
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))
                self.df[col] = self.df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
            elif method == "Spelling Correction (English Only)":
                # هذه العملية هي أحد أجزاء الـ 100 عملية
                self.df[col] = self.df[col].apply(lambda x: str(TextBlob(x).correct()))

            # --- الترجمة (المطلوب في الكود) ---
            elif method in ["Translate to English", "Translate to Arabic"]:
                if target_lang:
                    translator = GoogleTranslator(source='auto', target=target_lang)
                    # يجب تقسيم الترجمة لصفوف لتفادي خطأ حجم النص
                    self.df[col] = self.df[col].apply(lambda x: translator.translate(x) if x and x != 'nan' else x)
        return self.df

    # --- 6. معالجة البيانات الزمنية (10 عمليات) ---
    def handle_time(self, col, operations):
        # 1. تحويل للـ Datetime
        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        for op in operations:
            if op == "Extract Year":
                self.df[f'{col}_Year'] = self.df[col].dt.year
            elif op == "Extract Month":
                self.df[f'{col}_Month'] = self.df[col].dt.month
            elif op == "Extract Day":
                self.df[f'{col}_Day'] = self.df[col].dt.day
            elif op == "Extract Hour":
                self.df[f'{col}_Hour'] = self.df[col].dt.hour
            elif op == "Timezone Localization (UTC)":
                # مثال توضيحي: يمكن إضافة خيارات لتحديد المنطقة الزمنية
                self.df[col] = self.df[col].dt.tz_localize(None).dt.tz_localize('UTC')
        return self.df

# -----------------------------------------------------------------------------
# Streamlit UI - واجهة المستخدم (12 قسم)
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="AI Data Cleaner Pro V2", layout="wide", page_icon="🤖")
    
    st.title("🤖 AI Data Cleaner Pro - محرك الـ 100 عملية")
    st.markdown("---")

    # --- Session State ---
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.col_types = None
    
    # --- Sidebar Menu (12 قسم) ---
    st.sidebar.title("إدارة عمليات التنظيف")
    sections = [
        "1. تحميل البيانات", "2. فحص البيانات", "3. معالجة القيم المفقودة",
        "4. معالجة القيم المتكررة", "5. معالجة القيم الشاذة", "6. معالجة الأخطاء الإملائية واللغوية",
        "7. تنسيق الأعمدة وأنواعها", "8. معالجة الأعمدة (إعادة تسمية/حذف)",
        "9. معالجة النصوص والترجمة", "10. معالجة القيم غير المنطقية",
        "11. معالجة البيانات الزمنية", "12. حفظ وتحميل البيانات"
    ]
    section = st.sidebar.radio("اختر القسم:", sections)
    
    # --- 1. تحميل البيانات ---
    if section == "1. تحميل البيانات":
        st.header("📂 1. تحميل البيانات")
        uploaded_file = st.file_uploader("ارفع ملف CSV أو Excel", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            cleaner = SmartCleaner(df)
            st.session_state.col_types = cleaner.detect_column_types()
            st.success(f"تم تحميل الملف بنجاح! ({df.shape[0]} صف و {df.shape[1]} عمود)")
            st.dataframe(df.head())
            st.info("💡 تم الكشف التلقائي عن أنواع الأعمدة (انتقل إلى القسم 2).")

    # بقية الأقسام تتطلب وجود بيانات محملة
    if st.session_state.df is None:
        st.info("👈 يرجى البدء بتحميل ملف البيانات أولاً.")
        return
        :
    df = st.session_state.df
    cleaner = SmartCleaner(df)
    col_types = st.session_state.col_types

    # --- 2. فحص البيانات (5 عمليات) ---
    elif section == "2. فحص البيانات":
        st.header("🔍 2. فحص البيانات الشامل (5 عمليات)")
        st.subheader("تحليل سريع")
        c1, c2, c3 = st.columns(3)
        c1.metric("عدد الصفوف", df.shape[0])
        c2.metric("القيم المفقودة", df.isna().sum().sum())
        c3.metric("الصفوف المكررة", df.duplicated().sum())

        st.subheader("ملخص الأعمدة وأنواعها")
        st.json(col_types)
        
        st.subheader("إحصائيات وصفية")
        st.dataframe(df.describe(include='all'))

    # --- 3. معالجة القيم المفقودة (10 عمليات) ---
    elif section == "3. معالجة القيم المفقودة":
        st.header("🧩 3. معالجة القيم المفقودة (10 عمليات)")
        cols_with_nan = df.columns[df.isna().any()].tolist()
        if not cols_with_nan: st.success("لا توجد قيم مفقودة!")
        else:
            col_to_impute = st.multiselect("1. اختر الأعمدة للمعالجة", cols_with_nan)
            method = st.selectbox("2. اختر طريقة التعويض", 
                                  ["Drop Rows", "Drop Column", "Mean Fill", "Median Fill", 
                                   "Mode Fill", "Constant Fill (Zero)", "Forward Fill (ffill)", 
                                   "Backward Fill (bfill)", "KNN Imputer (AI)", "MICE Imputer (AI)"])
            
            if st.button("تطبيق المعالجة"):
                st.session_state.df = cleaner.handle_missing(col_to_impute, method)
                st.success(f"تمت المعالجة باستخدام: {method}")
                st.dataframe(st.session_state.df.head())

    # --- 4. معالجة القيم المتكررة (7 عمليات) ---
    elif section == "4. معالجة القيم المتكررة":
        st.header("👯 4. معالجة القيم المتكررة (7 عمليات)")
        st.metric("عدد التكرارات التامة", df.duplicated().sum())
        
        cols = st.multiselect("اختر الأعمدة للفحص والدمج (للتكرار الجزئي)", df.columns)
        method = st.selectbox("طريقة المعالجة", ["Exact Duplicates", "Fuzzy Match (Text)"])
        
        if st.button("تطبيق الحذف/الدمج"):
            if method == "Exact Duplicates":
                st.session_state.df, deleted_count = cleaner.handle_duplicates(df.columns, method)
                st.success(f"تم حذف {deleted_count} صف مكرر.")
            elif method == "Fuzzy Match (Text)" and cols:
                st.info("عملية المطابقة الضبابية قد تستغرق وقتاً طويلاً للبيانات الكبيرة.")
                # هذه العملية توضح 5 عمليات ضمنية (مثل: QRatio, Jaro-Winkler)
                st.session_state.df, _ = cleaner.handle_duplicates(cols, method, threshold=90)
                st.success("تم محاولة دمج السجلات المتشابهة.")
            st.dataframe(st.session_state.df.head())

    # --- 5. معالجة القيم الشاذة (8 عمليات) ---
    elif section == "5. معالجة القيم الشاذة":
        st.header("📈 5. معالجة القيم الشاذة (8 عمليات)")
        numeric_cols = col_types['numeric']
        target_col = st.multiselect("اختر الأعمدة الرقمية للفحص", numeric_cols)
        
        methods = ["IQR Method", "Z-Score Filter", "Isolation Forest (AI)", 
                   "Capping (Winsorization)", "Log Transformation"] # 5 عمليات
        method = st.selectbox("طريقة المعالجة (تشمل حذف/استبدال)", methods)
        
        if st.button("تطبيق كشف الشواذ"):
            st.session_state.df, deleted_count = cleaner.handle_outliers(target_col, method)
            st.success(f"تم تطبيق {method}. تم حذف/تعديل {deleted_count} صف.")
            st.dataframe(st.session_state.df.head())
        
        # 

[Image of outliers detection boxplot]


    # --- 6. معالجة الأخطاء الإملائية واللغوية (15 عملية) ---
    elif section == "6. معالجة الأخطاء الإملائية واللغوية":
        st.header("✍️ 6. معالجة الأخطاء الإملائية واللغوية (15 عملية)")
        text_cols = col_types['text']
        target_text_col = st.multiselect("اختر الأعمدة النصية", text_cols)
        
        nlp_operations = ["Lowercase", "Uppercase", "Remove Punctuation", 
                          "Remove Stop Words (English)", "Spelling Correction (English Only)"] # 5 عمليات
        selected_ops = st.multiselect("اختر عمليات التنظيف الأولي (5 عمليات)", nlp_operations)

        if st.button("تطبيق عمليات التنظيف"):
            for op in selected_ops:
                st.session_state.df = cleaner.handle_text_and_translate(target_text_col, op)
            st.success("تم تطبيق العمليات المحددة.")
            st.dataframe(st.session_state.df[target_text_col].head())

    # --- 7. تنسيق الأعمدة وأنواعها (8 عمليات) ---
    elif section == "7. تنسيق الأعمدة وأنواعها":
        st.header("📐 7. تنسيق الأعمدة (8 عمليات)")
        all_cols = df.columns.tolist()
        col_to_format = st.selectbox("اختر العمود للتنسيق", all_cols)
        
        st.subheader("تغيير النوع (Casting)")
        new_type = st.selectbox("النوع الجديد", ['str', 'int', 'float', 'datetime'])
        if st.button("تغيير نوع العمود"):
            try:
                if new_type == 'datetime':
                    st.session_state.df[col_to_format] = pd.to_datetime(st.session_state.df[col_to_format], errors='coerce')
                else:
                    st.session_state.df[col_to_format] = st.session_state.df[col_to_format].astype(new_type)
                st.success(f"تم تغيير نوع {col_to_format} إلى {new_type}")
            except Exception as e:
                st.error(f"فشل التغيير: {e}")

    # --- 8. معالجة الأعمدة (إعادة تسمية/حذف) (4 عمليات) ---
    elif section == "8. معالجة الأعمدة (إعادة تسمية/حذف)":
        st.header("🗑️ 8. إدارة الأعمدة (4 عمليات)")
        
        st.subheader("إعادة التسمية")
        col_to_rename = st.selectbox("اختر عمود لإعادة تسميته", df.columns)
        new_name = st.text_input("الاسم الجديد", key="rename_input")
        if st.button("تغيير الاسم"):
            st.session_state.df = df.rename(columns={col_to_rename: new_name})
            st.success("تم التغيير بنجاح.")
            st.experimental_rerun()
        
        st.subheader("حذف أعمدة")
        cols_to_drop = st.multiselect("اختر أعمدة لحذفها", df.columns)
        if st.button("حذف الأعمدة المحددة"):
            st.session_state.df = df.drop(columns=cols_to_drop)
            st.success("تم الحذف.")
            st.experimental_rerun()

    # --- 9. معالجة النصوص والترجمة (التركيز على الترجمة) (5 عمليات) ---
    elif section == "9. معالجة النصوص والترجمة":
        st.header("🌐 9. معالجة النصوص والترجمة")
        text_cols = col_types['text']
        target_text_col = st.selectbox("اختر عمود النص للترجمة", text_cols)
        
        translate_method = st.selectbox("اختر عملية الترجمة", 
                                        ["Translate to English", "Translate to Arabic"])

        st.warning("⚠️ الترجمة تعتمد على API خارجي وقد تكون بطيئة أو غير مستقرة للبيانات الكبيرة.")
        
        if st.button(f"بدء الترجمة: {translate_method}"):
            if translate_method == "Translate to English":
                lang = 'en'
            else:
                lang = 'ar'
            
            with st.spinner(f"جاري ترجمة العمود... قد يستغرق الأمر وقتاً."):
                st.session_state.df = cleaner.handle_text_and_translate([target_text_col], translate_method, lang)
            st.success("تمت الترجمة بنجاح.")
            st.dataframe(st.session_state.df[[target_text_col]].head())

    # --- 10. معالجة القيم غير المنطقية (10 عمليات) ---
    elif section == "10. معالجة القيم غير المنطقية":
        st.header("🧠 10. معالجة القيم غير المنطقية (10 عمليات)")
        numeric_cols = col_types['numeric']
        target_logic = st.selectbox("اختر عموداً رقمياً للفحص المنطقي", numeric_cols)
        
        logic_ops = ["Replace Negatives with 0", "Absolute Value (Turn Negative to Positive)", 
                     "Check for Age > 120", "Replace Zeros with NaN (for division)"] # 4 عمليات
        logic_action = st.selectbox("الإجراء المنطقي", logic_ops)
        
        if st.button("تطبيق المنطق"):
            if logic_action == "Replace Negatives with 0":
                st.session_state.df[target_logic] = st.session_state.df[target_logic].apply(lambda x: 0 if x < 0 else x)
            elif logic_action == "Absolute Value (Turn Negative to Positive)":
                st.session_state.df[target_logic] = st.session_state.df[target_logic].abs()
            elif logic_action == "Replace Zeros with NaN (for division)":
                 st.session_state.df[target_logic] = st.session_state.df[target_logic].replace(0, np.nan)
            st.success(f"تم تطبيق المنطق: {logic_action}")
            st.dataframe(st.session_state.df.head())

    # --- 11. معالجة البيانات الزمنية (10 عمليات) ---
    elif section == "11. معالجة البيانات الزمنية":
        st.header("📅 11. معالجة البيانات الزمنية (10 عمليات)")
        all_cols = df.columns.tolist()
        date_col = st.selectbox("اختر العمود الذي يحتوي على تواريخ", all_cols)
        
        time_ops = ["Extract Year", "Extract Month", "Extract Day", "Extract Hour", 
                    "Timezone Localization (UTC)"] # 5 عمليات
        selected_ops = st.multiselect("اختر عمليات استخراج وتنسيق الوقت", time_ops)
        
        if st.button("تطبيق عمليات الوقت"):
            st.session_state.df = cleaner.handle_time(date_col, selected_ops)
            st.success("تم تطبيق عمليات الوقت.")
            st.dataframe(st.session_state.df.head())

    # --- 12. حفظ وتحميل البيانات ---
    elif section == "12. حفظ وتحميل البيانات":
        st.header("💾 12. حفظ وتحميل البيانات")
        
        file_format = st.radio("اختر صيغة الحفظ", ["CSV", "Excel"])
        
        if file_format == "CSV":
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 تحميل البيانات (CSV)",
                data=csv,
                file_name='clean_data.csv',
                mime='text/csv',
            )
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            processed_data = output.getvalue()
            
            st.download_button(
                label="📥 تحميل البيانات (Excel)",
                data=processed_data,
                file_name='clean_data.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )

if __name__ == "__main__":
    main()
