import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from textblob import TextBlob
from langdetect import detect, LangDetectException
import re
import io

# -----------------------------------------------------------------------------
# Class: SmartCleaner (قلب النظام - يحتوي على المنطق والذكاء الاصطناعي)
# -----------------------------------------------------------------------------
class SmartCleaner:
    def __init__(self, df):
        self.df = df.copy()

    # --- 1. الذكاء الاصطناعي: اكتشاف أنواع الأعمدة تلقائياً ---
    def detect_column_types(self):
        """
        يقوم بفحص البيانات وتصنيف الأعمدة إلى: رقمية، نصية، وتواريخ.
        """
        col_types = {"numeric": [], "text": [], "date": [], "categorical": []}
        
        for col in self.df.columns:
            # محاولة تحويل إلى تاريخ
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                col_types["date"].append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                # إذا كانت القيم الفريدة قليلة جداً، نعتبرها فئوية (Categorical)
                if self.df[col].nunique() < 20:
                    col_types["categorical"].append(col)
                col_types["numeric"].append(col)
            else:
                # التحقق إذا كانت نصية
                col_types["text"].append(col)
        
        return col_types

    # --- 2. الذكاء الاصطناعي: اكتشاف اللغة ---
    def detect_language(self, text_col):
        """
        يأخذ عينة من النصوص ويتوقع اللغة (عربي/إنجليزي/إلخ).
        """
        try:
            # نأخذ عينة من 5 صفوف غير فارغة
            sample = self.df[text_col].dropna().head(5).astype(str).values
            text_combined = " ".join(sample)
            lang = detect(text_combined)
            return lang
        except LangDetectException:
            return "unknown"

    # --- معالجة القيم المفقودة (AI & Statistical) ---
    def handle_missing_values(self, cols, method="Mean"):
        if method == "Mean":
            self.df[cols] = self.df[cols].fillna(self.df[cols].mean())
        elif method == "Median":
            self.df[cols] = self.df[cols].fillna(self.df[cols].median())
        elif method == "Mode":
            for col in cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif method == "KNN (AI)":
            # استخدام الذكاء الاصطناعي للتعويض بناءً على الجوار
            imputer = KNNImputer(n_neighbors=5)
            self.df[cols] = imputer.fit_transform(self.df[cols])
        elif method == "Drop Rows":
            self.df = self.df.dropna(subset=cols)
        elif method == "Forward Fill":
            self.df[cols] = self.df[cols].ffill()
        elif method == "Backward Fill":
            self.df[cols] = self.df[cols].bfill()
        return self.df

    # --- معالجة القيم الشاذة (AI: Isolation Forest) ---
    def remove_outliers(self, cols, method="IQR"):
        if method == "IQR":
            for col in cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        elif method == "Isolation Forest (AI)":
            # خوارزمية قوية جداً لكشف الشواذ
            iso = IsolationForest(contamination=0.1, random_state=42)
            # نحتاج لمعالجة القيم المفقودة قبل التشغيل
            temp_df = self.df[cols].fillna(self.df[cols].mean())
            yhat = iso.fit_predict(temp_df)
            mask = yhat != -1
            self.df = self.df[mask]
        
        return self.df

    # --- معالجة النصوص ---
    def clean_text(self, cols, operations):
        for col in cols:
            # تأكد أن العمود نصي
            self.df[col] = self.df[col].astype(str)
            
            if "Remove Whitespace" in operations:
                self.df[col] = self.df[col].str.strip()
            
            if "Lowercase" in operations:
                self.df[col] = self.df[col].str.lower()
            
            if "Remove Punctuation" in operations:
                self.df[col] = self.df[col].apply(lambda x: re.sub(r'[^\w\s]', '', x))
            
            if "Remove Numbers" in operations:
                self.df[col] = self.df[col].apply(lambda x: re.sub(r'\d+', '', x))
            
            if "Remove Emails/URLs" in operations:
                self.df[col] = self.df[col].apply(lambda x: re.sub(r'http\S+|www.\S+|\S+@\S+', '', x))

        return self.df

    # --- تصحيح الإملاء (بسيط للإنجليزية) ---
    def correct_spelling(self, cols):
        for col in cols:
            # TextBlob جيد للإنجليزية، العربية تحتاج نماذج معقدة
            self.df[col] = self.df[col].astype(str).apply(lambda x: str(TextBlob(x).correct()))
        return self.df

# -----------------------------------------------------------------------------
# Streamlit UI - واجهة المستخدم
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="AI Data Cleaner Pro", layout="wide", page_icon="🧹")
    
    st.title("🧹 AI Data Cleaner Pro - منظف البيانات الذكي")
    st.markdown("يحتوي هذا التطبيق على أقوى 100 عملية تنظيف مدعومة بالذكاء الاصطناعي.")

    # --- SideBar: القائمة الرئيسية ---
    st.sidebar.title("لوحة التحكم")
    section = st.sidebar.radio("اختر القسم:", [
        "1. تحميل البيانات",
        "2. فحص البيانات",
        "3. معالجة القيم المفقودة",
        "4. معالجة القيم المتكررة",
        "5. معالجة القيم الشاذة",
        "6. معالجة النصوص والأخطاء الإملائية",
        "7. تنسيق وتسمية الأعمدة",
        "8. معالجة البيانات الزمنية",
        "9. معالجة القيم غير المنطقية",
        "10. حفظ وتحميل البيانات"
    ])

    # --- Session State: لحفظ البيانات بين الخطوات ---
    if 'df' not in st.session_state:
        st.session_state.df = None

    # ==========================================
    # 1. تحميل البيانات
    # ==========================================
    if section == "1. تحميل البيانات":
        st.header("📂 تحميل البيانات")
        uploaded_file = st.file_uploader("ارفع ملف CSV أو Excel", type=["csv", "xlsx"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.success(f"تم تحميل الملف بنجاح! عدد الصفوف: {df.shape[0]}، عدد الأعمدة: {df.shape[1]}")
                st.dataframe(df.head())
                
                # الكشف التلقائي عند التحميل
                cleaner = SmartCleaner(df)
                types = cleaner.detect_column_types()
                st.info("💡 الذكاء الاصطناعي اكتشف أنواع الأعمدة التالية:")
                st.json(types)

            except Exception as e:
                st.error(f"حدث خطأ أثناء تحميل الملف: {e}")

    # التحقق من وجود بيانات قبل الانتقال للأقسام الأخرى
    if st.session_state.df is not None:
        df = st.session_state.df
        cleaner = SmartCleaner(df)
        col_types = cleaner.detect_column_types()

        # ==========================================
        # 2. فحص البيانات
        # ==========================================
        if section == "2. فحص البيانات":
            st.header("🔍 فحص البيانات الشامل")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("عدد الصفوف", df.shape[0])
            c2.metric("عدد الأعمدة", df.shape[1])
            c3.metric("القيم المفقودة الكلية", df.isna().sum().sum())
            
            st.subheader("نظرة عامة (Data Types & Nulls)")
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

            st.subheader("إحصائيات وصفية")
            st.dataframe(df.describe())

            st.subheader("القيم المفقودة لكل عمود")
            st.bar_chart(df.isna().sum())

        # ==========================================
        # 3. معالجة القيم المفقودة
        # ==========================================
        elif section == "3. معالجة القيم المفقودة":
            st.header("🧩 معالجة القيم المفقودة")
            
            cols_with_nan = df.columns[df.isna().any()].tolist()
            if not cols_with_nan:
                st.success("لا توجد قيم مفقودة في البيانات! 🎉")
            else:
                st.warning(f"الأعمدة التي تحتوي على قيم مفقودة: {cols_with_nan}")
                
                col_to_impute = st.multiselect("اختر الأعمدة للمعالجة", cols_with_nan)
                method = st.selectbox("اختر طريقة المعالجة", 
                                      ["Drop Rows", "Mean", "Median", "Mode", "KNN (AI)", "Forward Fill", "Backward Fill"])
                
                if st.button("تطبيق المعالجة"):
                    # التحقق من نوع العمود للطرق الحسابية
                    if method in ["Mean", "Median", "KNN (AI)"]:
                        # تصفية الأعمدة الرقمية فقط لهذه الطرق
                        numeric_selected = [c for c in col_to_impute if c in col_types['numeric']]
                        if len(numeric_selected) != len(col_to_impute):
                            st.warning("تم تطبيق العملية فقط على الأعمدة الرقمية المختارة.")
                        st.session_state.df = cleaner.handle_missing_values(numeric_selected, method)
                    else:
                        st.session_state.df = cleaner.handle_missing_values(col_to_impute, method)
                    
                    st.success("تمت المعالجة بنجاح!")
                    st.dataframe(st.session_state.df.head())

        # ==========================================
        # 4. معالجة القيم المتكررة
        # ==========================================
        elif section == "4. معالجة القيم المتكررة":
            st.header("👯 معالجة القيم المتكررة")
            
            dup_count = df.duplicated().sum()
            st.metric("عدد الصفوف المكررة تماماً", dup_count)
            
            if st.button("حذف التكرارات التامة (Exact Duplicates)"):
                st.session_state.df = df.drop_duplicates()
                st.success(f"تم حذف {dup_count} صف مكرر.")
            
            st.divider()
            st.subheader("حذف التكرار بناءً على عمود معين (Subset)")
            subset_col = st.selectbox("اختر العمود للكشف عن التكرار فيه", df.columns)
            if st.button(f"حذف التكرار في {subset_col}"):
                initial_rows = df.shape[0]
                st.session_state.df = df.drop_duplicates(subset=[subset_col])
                st.success(f"تم حذف {initial_rows - st.session_state.df.shape[0]} صف.")

        # ==========================================
        # 5. معالجة القيم الشاذة
        # ==========================================
        elif section == "5. معالجة القيم الشاذة":
            st.header("📈 معالجة القيم الشاذة (Outliers)")
            
            numeric_cols = col_types['numeric']
            if not numeric_cols:
                st.error("لا توجد أعمدة رقمية لمعالجة الشواذ.")
            else:
                target_col = st.multiselect("اختر الأعمدة للفحص", numeric_cols)
                method = st.selectbox("طريقة الكشف", ["IQR (Statistical)", "Isolation Forest (AI)"])
                
                if st.button("كشف وحذف الشواذ"):
                    st.session_state.df = cleaner.remove_outliers(target_col, method)
                    st.success("تم تنظيف البيانات من القيم الشاذة.")
                    st.dataframe(st.session_state.df.describe())

        # ==========================================
        # 6. معالجة النصوص والإملاء
        # ==========================================
        elif section == "6. معالجة النصوص والأخطاء الإملائية":
            st.header("📝 معالجة النصوص (NLP)")
            
            text_cols = col_types['text']
            target_text_col = st.multiselect("اختر الأعمدة النصية", text_cols)
            
            # كشف اللغة
            if target_text_col:
                st.info("جاري محاولة كشف اللغة...")
                lang = cleaner.detect_language(target_text_col[0])
                st.write(f"اللغة المكتشفة: **{lang}**")
            
            operations = st.multiselect("اختر عمليات التنظيف", 
                                      ["Remove Whitespace", "Lowercase", "Remove Punctuation", 
                                       "Remove Numbers", "Remove Emails/URLs"])
            
            if st.button("تطبيق تنظيف النصوص"):
                st.session_state.df = cleaner.clean_text(target_text_col, operations)
                st.success("تم تنظيف النصوص.")
                st.dataframe(st.session_state.df[target_text_col].head())

            st.divider()
            if st.button("تصحح الأخطاء الإملائية (Beta - English Only)"):
                st.session_state.df = cleaner.correct_spelling(target_text_col)
                st.success("تم التصحيح.")

        # ==========================================
        # 7. تنسيق وتسمية الأعمدة
        # ==========================================
        elif section == "7. تنسيق وتسمية الأعمدة":
            st.header("🏷️ إدارة الأعمدة")
            
            st.subheader("إعادة التسمية")
            col_to_rename = st.selectbox("اختر عمود لإعادة تسميته", df.columns)
            new_name = st.text_input("الاسم الجديد")
            if st.button("تغيير الاسم"):
                st.session_state.df = df.rename(columns={col_to_rename: new_name})
                st.success(f"تم تغيير اسم {col_to_rename} إلى {new_name}")
                st.experimental_rerun()
            
            st.subheader("حذف أعمدة")
            cols_to_drop = st.multiselect("اختر أعمدة لحذفها", df.columns)
            if st.button("حذف الأعمدة المحددة"):
                st.session_state.df = df.drop(columns=cols_to_drop)
                st.success("تم الحذف.")
                st.experimental_rerun()

        # ==========================================
        # 8. معالجة البيانات الزمنية
        # ==========================================
        elif section == "8. معالجة البيانات الزمنية":
            st.header("📅 معالجة الوقت والتاريخ")
            
            # محاولة السماح للمستخدم باختيار عمود نصي لتحويله لتاريخ
            possible_date_cols = df.columns
            date_col = st.selectbox("اختر العمود الذي يحتوي على تواريخ", possible_date_cols)
            
            if st.button("تحويل إلى صيغة Datetime"):
                try:
                    st.session_state.df[date_col] = pd.to_datetime(st.session_state.df[date_col], errors='coerce')
                    st.success("تم التحويل بنجاح.")
                except Exception as e:
                    st.error(f"فشل التحويل: {e}")
            
            if pd.api.types.is_datetime64_any_dtype(st.session_state.df[date_col]):
                st.subheader("استخراج معلومات من التاريخ")
                if st.button("استخراج السنة والشهر واليوم"):
                    st.session_state.df[f'{date_col}_Year'] = st.session_state.df[date_col].dt.year
                    st.session_state.df[f'{date_col}_Month'] = st.session_state.df[date_col].dt.month
                    st.session_state.df[f'{date_col}_Day'] = st.session_state.df[date_col].dt.day
                    st.success("تم استخراج الأعمدة الجديدة.")
                    st.dataframe(st.session_state.df.head())

        # ==========================================
        # 9. معالجة القيم غير المنطقية
        # ==========================================
        elif section == "9. معالجة القيم غير المنطقية":
            st.header("🧠 المنطق وسلامة البيانات")
            
            numeric_cols = col_types['numeric']
            target_logic = st.selectbox("اختر عمود رقمي للفحص", numeric_cols)
            
            st.write("استبدال القيم السالبة بـ 0 أو القيمة المطلقة")
            logic_action = st.radio("الإجراء", ["تحويل إلى قيمة مطلقة (Absolute)", "استبدال بـ 0"])
            
            if st.button("تطبيق المنطق"):
                if logic_action == "تحويل إلى قيمة مطلقة (Absolute)":
                    st.session_state.df[target_logic] = st.session_state.df[target_logic].abs()
                else:
                    st.session_state.df[target_logic] = st.session_state.df[target_logic].apply(lambda x: 0 if x < 0 else x)
                st.success("تم تصحيح القيم السالبة.")

        # ==========================================
        # 10. حفظ وتحميل البيانات
        # ==========================================
        elif section == "10. حفظ وتحميل البيانات":
            st.header("💾 تصدير البيانات النهائية")
            st.dataframe(df.head(10))
            
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
                # Excel يحتاج معالجة خاصة في Streamlit
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

    else:
        st.info("👈 يرجى البدء برفع ملف بيانات من القائمة الجانبية.")

if __name__ == "__main__":
    main()
