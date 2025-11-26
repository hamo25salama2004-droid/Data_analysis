import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from deep_translator import GoogleTranslator
import plotly.express as px # تم إضافة الاستيراد هنا لضمان عمل قسم تحليل الربحية

# ==========================================
# إعداد الصفحة
# ==========================================
st.set_page_config(page_title="أداة تنظيف البيانات المتكاملة", layout="wide", page_icon="📊")

# ------------------------------------------------------------------
# إدارة حالة الجلسة (Session State)
# ------------------------------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None

# ------------------------------------------------------------------
# دوال مساعدة
# ------------------------------------------------------------------
def convert_df(df, file_type):
    """تحويل الداتا فريم إلى ملف بايت للتحميل"""
    buffer = BytesIO()
    if file_type == 'csv':
        # استخدام utf-8-sig لدعم اللغة العربية في ملفات CSV المحملة على ويندوز/إكسل
        df.to_csv(buffer, index=False, encoding='utf-8-sig')
    else:
        df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer

# ------------------------------------------------------------------
# القائمة الجانبية (Sidebar)
# ------------------------------------------------------------------
st.sidebar.title("لوحة التحكم")
st.sidebar.markdown("---")

options = [
    "تحميل البيانات",
    "فحص البيانات",
    "معالجة القيم المفقودة",
    "معالجة القيم المتكررة",
    "معالجة القيم الشاذة",
    "معالجة الأخطاء الإملائية",
    "تنسيق الأعمدة",
    "معالجة الأعمدة (إعادة تسمية/حذف)",
    "معالجة النصوص والترجمة",
    "معالجة القيم غير المنطقية",
    "معالجة البيانات الزمنية",
    "تحليل الربحية", # تم التأكد من إضافة الخيار هنا
    "حفظ وتحميل البيانات"
]

choice = st.sidebar.radio("اختر القسم:", options)

st.title("🛠️ أداة تنظيف وتحليل البيانات الشاملة")

# ------------------------------------------------------------------
# 1. تحميل البيانات
# ------------------------------------------------------------------
if choice == "تحميل البيانات":
    st.header("📂 تحميل ملف البيانات")
    uploaded_file = st.file_uploader("اختر ملف (CSV أو Excel)", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                # محاولة القراءة بإعدادات مختلفة إذا فشلت الأولى
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin1')
            else:
                df = pd.read_excel(uploaded_file)
                
            st.session_state.df = df
            st.success(f"تم تحميل الملف '{uploaded_file.name}' بنجاح!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحميل: {e}")

# التحقق من وجود بيانات
if st.session_state.df is not None:
    df = st.session_state.df # اختصار

    # ------------------------------------------------------------------
    # 2. فحص البيانات
    # ------------------------------------------------------------------
    if choice == "فحص البيانات":
        st.header("🔍 فحص البيانات")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"عدد الصفوف: {df.shape[0]}")
        with col2:
            st.info(f"عدد الأعمدة: {df.shape[1]}")

        st.subheader("أنواع البيانات")
        st.write(df.dtypes.astype(str))
        
        st.subheader("إحصائيات وصفية")
        st.write(df.describe(include='all'))

    # ------------------------------------------------------------------
    # 3. معالجة القيم المفقودة
    # ------------------------------------------------------------------
    elif choice == "معالجة القيم المفقودة":
        st.header("🧩 معالجة القيم المفقودة")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.warning("يوجد قيم مفقودة:")
            st.write(missing_data[missing_data > 0])
            
            action = st.selectbox("اختر إجراء:", ["حذف الصفوف", "حذف الأعمدة", "تعويض القيم"])
            
            if action == "حذف الصفوف":
                if st.button("تطبيق"):
                    st.session_state.df = df.dropna(axis=0)
                    st.success("تم الحذف.")
                    st.rerun()
            elif action == "حذف الأعمدة":
                if st.button("تطبيق"):
                    st.session_state.df = df.dropna(axis=1)
                    st.success("تم الحذف.")
                    st.rerun()
            elif action == "تعويض القيم":
                col_to_fill = st.selectbox("العمود:", df.columns)
                method = st.radio("الطريقة:", ["المتوسط", "الوسيط", "الوضع", "قيمة ثابتة"])
                val_to_fill = st.text_input("القيمة الثابتة:") if method == "قيمة ثابتة" else None

                if st.button("تطبيق"):
                    try:
                        if method == "المتوسط": st.session_state.df[col_to_fill] = df[col_to_fill].fillna(df[col_to_fill].mean())
                        elif method == "الوسيط": st.session_state.df[col_to_fill] = df[col_to_fill].fillna(df[col_to_fill].median())
                        elif method == "الوضع": st.session_state.df[col_to_fill] = df[col_to_fill].fillna(df[col_to_fill].mode()[0])
                        elif method == "قيمة ثابتة": st.session_state.df[col_to_fill] = df[col_to_fill].fillna(val_to_fill)
                        st.success("تم التعويض.")
                        st.rerun()
                    except Exception as e: st.error(f"خطأ: {e}")
        else:
            st.success("لا توجد قيم مفقودة.")

    # ------------------------------------------------------------------
    # 4. معالجة القيم المتكررة
    # ------------------------------------------------------------------
    elif choice == "معالجة القيم المتكررة":
        st.header("👯 معالجة القيم المتكررة")
        duplicates = df.duplicated().sum()
        st.write(f"صفوف مكررة بالكامل: {duplicates}")
        if duplicates > 0 and st.button("حذف الكل"):
            st.session_state.df = df.drop_duplicates()
            st.success("تم الحذف.")
            st.rerun()

        st.divider()
        subset_cols = st.multiselect("حذف تكرار بناءً على أعمدة معينة:", df.columns)
        if subset_cols and st.button("حذف المحدد"):
            st.session_state.df = df.drop_duplicates(subset=subset_cols)
            st.success("تم الحذف.")
            st.rerun()

    # ------------------------------------------------------------------
    # 5. معالجة القيم الشاذة
    # ------------------------------------------------------------------
    elif choice == "معالجة القيم الشاذة":
        st.header("📈 معالجة القيم الشاذة")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("العمود الرقمي:", numeric_cols)
            method = st.radio("الطريقة:", ["IQR", "Z-Score"])
            
            if method == "IQR":
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            else:
                mean, std = df[col].mean(), df[col].std()
                lower, upper = mean - 3 * std, mean + 3 * std
            
            st.write(f"الحدود: {lower:.2f} - {upper:.2f}")
            outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
            st.write(f"عدد القيم الشاذة: {outliers}")
            
            if outliers > 0:
                act = st.selectbox("الإجراء:", ["حذف", "استبدال بالحدود"])
                if st.button("تطبيق"):
                    if act == "حذف": st.session_state.df = df[(df[col] >= lower) & (df[col] <= upper)]
                    else: st.session_state.df[col] = np.clip(df[col], lower, upper)
                    st.success("تم.")
                    st.rerun()
        else: st.warning("لا توجد أعمدة رقمية.")

    # ------------------------------------------------------------------
    # 6. معالجة الأخطاء الإملائية (تم تعديل هذا القسم)
    # ------------------------------------------------------------------
    elif choice == "معالجة الأخطاء الإملائية":
        st.header("📝 تنظيف النصوص")
        text_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(text_cols) > 0:
            col = st.selectbox("العمود:", text_cols)
            op = st.selectbox("العملية:", ["إزالة مسافات", "أحرف صغيرة", "أحرف كبيرة", "إزالة رموز خاصة"])
            
            # --- التعديل هنا: إضافة حقل إدخال لتحديد الرمز المراد حذفه ---
            regex_to_remove = None
            if op == "إزالة رموز خاصة":
                # القيمة الافتراضية هي ما كان عليه الكود الأصلي: حذف كل ما ليس حرفاً أو رقماً أو مسافة
                symbol_to_remove = st.text_input(
                    "الرمز أو الأنماط المراد حذفها (كـ تعبير عادي - Regex):", 
                    value=r'[^\w\s]', 
                    help="مثل: لإزالة علامات الترقيم فقط أدخل '[.,!?:;]'"
                )
                if symbol_to_remove:
                    regex_to_remove = symbol_to_remove
            # ----------------------------------------------------------

            if st.button("تطبيق"):
                st.session_state.df[col] = df[col].astype(str)
                if op == "إزالة مسافات": 
                    st.session_state.df[col] = df[col].str.strip()
                elif op == "أحرف صغيرة": 
                    st.session_state.df[col] = df[col].str.lower()
                elif op == "أحرف كبيرة": 
                    st.session_state.df[col] = df[col].str.upper()
                elif op == "إزالة رموز خاصة":
                    if regex_to_remove:
                        st.session_state.df[col] = df[col].apply(lambda x: re.sub(regex_to_remove, '', str(x)))
                    else:
                        st.warning("الرجاء إدخال الرمز أو الأنماط المراد حذفها.")
                        st.stop()
                st.success("تم.")
                st.rerun()
        else: st.warning("لا توجد أعمدة نصية.")

    # ------------------------------------------------------------------
    # 7. تنسيق الأعمدة
    # ------------------------------------------------------------------
    elif choice == "تنسيق الأعمدة":
        st.header("🔢 تنسيق الأعمدة")
        col = st.selectbox("العمود:", df.columns)
        to_type = st.selectbox("إلى:", ["رقمي", "تاريخ", "نص"])
        if st.button("تحويل"):
            try:
                if to_type == "رقمي": st.session_state.df[col] = pd.to_numeric(df[col], errors='coerce')
                elif to_type == "تاريخ": st.session_state.df[col] = pd.to_datetime(df[col], errors='coerce')
                else: st.session_state.df[col] = df[col].astype(str)
                st.success("تم التحويل.")
                st.rerun()
            except Exception as e: st.error(str(e))

    # ------------------------------------------------------------------
    # 8. معالجة الأعمدة
    # ------------------------------------------------------------------
    elif choice == "معالجة الأعمدة (إعادة تسمية/حذف)":
        st.header("🛠️ إدارة الأعمدة")
        tab1, tab2 = st.tabs(["إعادة تسمية", "حذف"])
        with tab1:
            old_name = st.selectbox("العمود القديم:", df.columns)
            new_name = st.text_input("الاسم الجديد:")
            if st.button("تغيير الاسم") and new_name:
                st.session_state.df = df.rename(columns={old_name: new_name})
                st.success("تم.")
                st.rerun()
        with tab2:
            drop_cols = st.multiselect("حذف أعمدة:", df.columns)
            if st.button("حذف") and drop_cols:
                st.session_state.df = df.drop(columns=drop_cols)
                st.success("تم.")
                st.rerun()

    # ------------------------------------------------------------------
    # 9. معالجة النصوص والترجمة
    # ------------------------------------------------------------------
    elif choice == "معالجة النصوص والترجمة":
        st.header("🔤 معالجة النصوص المتقدمة والترجمة")
        text_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(text_cols) > 0:
            col = st.selectbox("العمود النصي:", text_cols)
            task = st.radio("المهمة:", ["إزالة الأرقام", "ترجمة (عربي <> إنجليزي)"])
            
            if task == "إزالة الأرقام":
                if st.button("تطبيق"):
                    st.session_state.df[col] = df[col].astype(str).apply(lambda x: re.sub(r'\d+', '', x))
                    st.success("تم.")
                    st.rerun()
            
            elif task == "ترجمة (عربي <> إنجليزي)":
                st.markdown("### 🌍 الترجمة الفورية")
                trans_dir = st.selectbox("الاتجاه:", ["من الإنجليزية إلى العربية", "من العربية إلى الإنجليزية"])
                
                if st.button("بدء الترجمة (قد يستغرق وقتاً)"):
                    try:
                        src = 'en' if "الإنجليزية إلى العربية" in trans_dir else 'ar'
                        dest = 'ar' if "الإنجليزية إلى العربية" in trans_dir else 'en'
                        translator = GoogleTranslator(source=src, target=dest)
                        
                        prog = st.progress(0)
                        res_list = []
                        total = len(df)
                        
                        for i, txt in enumerate(df[col].astype(str)):
                            if txt and txt.strip() and txt.lower() != 'nan':
                                try:
                                    res_list.append(translator.translate(txt))
                                except:
                                    res_list.append(txt)
                            else:
                                res_list.append(txt)
                            if i % 5 == 0: prog.progress((i+1)/total)
                        
                        prog.progress(1.0)
                        st.session_state.df[col] = res_list
                        st.success("تمت الترجمة!")
                        st.rerun()
                    except Exception as e: st.error(f"خطأ: {e}")
        else: st.warning("لا توجد أعمدة نصية.")

    # ------------------------------------------------------------------
    # 10. معالجة القيم غير المنطقية
    # ------------------------------------------------------------------
    elif choice == "معالجة القيم غير المنطقية":
        st.header("🤔 استبدال القيم")
        col = st.selectbox("العمود:", df.columns, key='ill_col')
        v_old = st.text_input("القيمة القديمة:")
        v_new = st.text_input("القيمة الجديدة (فراغ = NaN):")
        if st.button("استبدال"):
            val = v_new if v_new else np.nan
            st.session_state.df[col] = df[col].replace(v_old, val) # قد يحتاج ضبط أنواع
            st.success("تم.")
            st.rerun()

    # ------------------------------------------------------------------
    # 11. البيانات الزمنية
    # ------------------------------------------------------------------
    elif choice == "معالجة البيانات الزمنية":
        st.header("📅 السلاسل الزمنية")
        d_col = st.selectbox("عمود التاريخ:", df.columns)
        if st.button("تحويل لفهرس زمني"):
            try:
                st.session_state.df[d_col] = pd.to_datetime(df[d_col], errors='coerce')
                st.session_state.df = st.session_state.df.dropna(subset=[d_col]).set_index(d_col).sort_index()
                st.success("تم.")
                st.rerun()
            except: st.error("فشل التحويل.")
        if isinstance(df.index, pd.DatetimeIndex) and st.button("إلغاء الفهرس الزمني"):
            st.session_state.df = df.reset_index()
            st.rerun()

    # ------------------------------------------------------------------
    # 12. تحليل الربحية
    # ------------------------------------------------------------------
    elif choice == "تحليل الربحية":
        # تم نقل استيراد plotly.express إلى أعلى الكود لضمان توفره
        st.header("💰 تحليل الربحية (Profit Analysis)")
        
        # التأكد من وجود أعمدة رقمية قابلة للحساب
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        col_profit = st.selectbox("اختر عمود المبيعات/الإيرادات:", numeric_cols if numeric_cols else [None], key="profit_col")
        
        # إضافة خيار None للتكلفة لتمثيل حالة عدم وجود عمود تكلفة
        cost_options = [None] + numeric_cols
        col_cost = st.selectbox("اختر عمود التكلفة (اختياري):", cost_options, key="cost_col")
        
        col_product = st.selectbox("اختر عمود المنتج/الفئة للتجميع:", df.columns, key="prod_col")

        # التأكد من أن المستخدم اختار الأعمدة المطلوبة قبل الحساب
        if col_profit and col_product:
            
            # محاولة تحويل الأعمدة إلى رقمية مرة أخرى لضمان عدم وجود مشاكل في Dtypes
            try:
                temp_df = df.copy()
                temp_df[col_profit] = pd.to_numeric(temp_df[col_profit], errors='coerce')
                
                # حساب صافي الربح
                if col_cost and col_cost != None:
                    temp_df[col_cost] = pd.to_numeric(temp_df[col_cost], errors='coerce')
                    temp_df['Net Profit'] = temp_df[col_profit] - temp_df[col_cost]
                    profit_column_name = 'Net Profit'
                    title_text = "صافي الربح لكل فئة/منتج"
                else:
                    profit_column_name = col_profit
                    title_text = "إجمالي المبيعات/الإيرادات لكل فئة/منتج"
                    
                # التجميع والرسم
                grouped_data = temp_df.groupby(col_product)[profit_column_name].sum().reset_index()
                
                profit_fig = px.bar(
                    grouped_data, 
                    x=col_product, 
                    y=profit_column_name, 
                    title=title_text,
                    labels={col_product: col_product, profit_column_name: profit_column_name}
                )
                st.plotly_chart(profit_fig, use_container_width=True)
                st.success("تم عرض التحليل بنجاح.")
                
            except Exception as e:
                st.error(f"حدث خطأ في الحساب أو الرسم البياني. تأكد أن أعمدة المبيعات والتكلفة رقمية. التفاصيل: {e}")
        else:
            st.warning("الرجاء اختيار عمود المبيعات وعمود المنتج/الفئة على الأقل للتحليل.")


    # ------------------------------------------------------------------
    # 13. حفظ وتحميل
    # ------------------------------------------------------------------
    elif choice == "حفظ وتحميل البيانات":
        st.header("💾 تحميل النتائج")
        st.dataframe(df.head())
        fn = st.text_input("اسم الملف:", "data_cleaned")
        
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("تحميل CSV", convert_df(df, 'csv'), f"{fn}.csv", "text/csv")
        with c2:
            st.download_button("تحميل Excel", convert_df(df, 'excel'), f"{fn}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    if choice != "تحميل البيانات": st.info("الرجاء تحميل ملف أولاً.")
