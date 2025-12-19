import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURARE PAGINA ---
st.set_page_config(page_title="Tema EDA - Analiza Volumelor Mari", layout="wide")

# Paleta de culori unificata
PERIWINKLE = "#8587D6"
PEACH = "#ffaaa5"

st.markdown(f"""
<style>
    .main-header {{ font-size: 2.5rem; color: {PERIWINKLE}; text-align: center; margin-bottom: 2rem; font-weight: 700; }}
    .sub-header {{ font-size: 1.5rem; color: {PEACH}; font-weight: 600; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid {PERIWINKLE}; padding-bottom: 0.5rem; }}
    
    .success-box-green {{ 
        background-color: #d4edda; 
        padding: 1rem; 
        border-radius: 10px; 
        border-left: 5px solid #28a745; 
        color: #155724;
        font-size: 0.9rem;
        margin: 1rem 0;
    }}

    .stSlider [data-baseweb="slider"] > div > div {{ background-color: {PEACH} !important; }}
    .stSlider [data-baseweb="slider"] div[role="slider"] {{ background-color: {PERIWINKLE} !important; border: 2px solid {PEACH} !important; }}
    .stMultiSelect [data-baseweb="tag"] {{ background-color: {PERIWINKLE} !important; }}
    .desc-section {{ line-height: 1.6; font-size: 1.05rem; margin-bottom: 1.2rem; text-align: justify; }}
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state: st.session_state['df'] = None
if 'filename' not in st.session_state: st.session_state['filename'] = ""

# --- SIDEBAR ---
def sidebar_navigation():
    st.sidebar.markdown(f"<h1 style='color:{PERIWINKLE};'>Tema EDA</h1>", unsafe_allow_html=True)
    sections = [
        "Descriere Aplicație",
        "Cerinta 1: Incarcare si Filtrare", 
        "Cerinta 2: Structura si Valori Lipsa", 
        "Cerinta 3: Analiza Univariata", 
        "Cerinta 4: Analiza Categorica", 
        "Cerinta 5: Corelatie si Outlieri"
    ]
    return st.sidebar.radio("Navigare:", sections)

# --- PAGINA: DESCRIERE APLICATIE ---
def pagina_descriere():
    st.markdown('<h1 class="main-header">Descrierea Aplicației EDA cu Streamlit</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="desc-section">', unsafe_allow_html=True)
    st.write("Această aplicație a fost realizată folosind Streamlit și are ca scop efectuarea unei analize exploratorii a datelor (Exploratory Data Analysis – EDA) pe seturi de date de dimensiuni mari, încărcate de utilizator sub formă de fișiere CSV sau Excel. Aplicația permite explorarea interactivă a datelor fără a necesita cunoștințe avansate de programare, oferind un mediu vizual intuitiv pentru inspectarea structurii datasetului, identificarea valorilor lipsă, analiza distribuțiilor, analiza variabilelor categorice, analiza relațiilor dintre variabile numerice și detecția valorilor extreme (outlieri). Structura aplicației este organizată pe cinci secțiuni principale, corespunzătoare cerințelor temei, fiecare secțiune fiind accesibilă din meniul lateral.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header">CERINȚA 1 – Încărcare și filtrare date</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-section">', unsafe_allow_html=True)
    st.write("În această secțiune, aplicația permite încărcarea unui fișier CSV sau Excel, care este validat automat pentru a verifica dacă a fost citit corect. După încărcare, se afișează un mesaj de confirmare care conține numărul de rânduri și coloane, iar primele 10 rânduri sunt prezentate pentru o previzualizare rapidă. Utilizatorul poate filtra datele folosind slidere interactive pentru coloanele numerice și meniuri multiselect pentru cele categorice. Aplicația afișează clar numărul de rânduri inițial față de cel rezultat după filtrare, facilitând izolarea rapidă a subseturilor relevante.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header">CERINȚA 2 – Structura datasetului și valori lipsă</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-section">', unsafe_allow_html=True)
    st.write("Această secțiune oferă o imagine de ansamblu asupra structurii datasetului, afișând numărul total de rânduri, coloane, memoria ocupată și procentul total de valori lipsă. Pentru fiecare coloană se prezintă tipul de date și numărul de valori prezente. Aplicația identifică automat coloanele cu probleme, calculează procentele de lipsă și afișează grafice dedicate, precum și statistici descriptive fundamentale, fiind o etapă esențială pentru evaluarea calității datelor înainte de analiză.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header">CERINȚA 3 – Analiză univariată</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-section">', unsafe_allow_html=True)
    st.write("Utilizatorul poate selecta aici orice coloană numerică pentru un studiu detaliat. Aplicația permite alegerea numărului de bins pentru histogramă printr-un slider, oferind o ilustrare vizuală a distribuției alături de un box plot. În paralel, sunt calculate și afișate metricele principale: media, mediana, deviația standard și valorile extreme. Această analiză ajută la înțelegerea formei distribuției și a variabilității datelor din coloana selectată.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header">CERINȚA 4 – Analiză categorică</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-section">', unsafe_allow_html=True)
    st.write("Secțiunea se concentrează pe variabilele discrete, identificând automat coloanele categorice. Aplicația generează un grafic de tip bar chart cu frecvențele fiecărei categorii și oferă un tabel detaliat care include frecvențele absolute și procentele corespunzătoare. Analiza este utilă pentru înțelegerea distribuției valorilor discrete și a eventualelor dezechilibre dintre categoriile prezente în setul de date.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header">CERINȚA 5 – Corelații și detecție outlieri</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-section">', unsafe_allow_html=True)
    st.write("Ultima secțiune explorează relațiile dintre variabilele numerice prin matricea de corelație și heatmap-uri interactive. Utilizatorul poate selecta două variabile pentru un scatter plot cu linie de regresie, calculând simultan coeficientul Pearson. În plus, aplicația utilizează metoda IQR pentru detecția valorilor extreme, afișând numărul și procentul de outlieri pentru fiecare coloană și vizualizându-i prin box plot-uri dedicate.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="desc-section">', unsafe_allow_html=True)
    st.write("În concluzie, aplicația dezvoltată oferă un instrument complet și interactiv pentru analiza exploratorie a datelor, respectând toate cerințele impuse în temă. Prin utilizarea Streamlit și Plotly, aceasta facilitează înțelegerea rapidă a dataseturilor și susține procesul de luare a deciziilor bazate pe date.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- SECTIUNE SEMNATURA ---
    st.markdown(f"""
        <div style="text-align: center; margin-top: 3rem; padding: 1rem; border-top: 1px solid #eee;">
            <p style="color: #666; font-size: 1.1rem; font-style: italic;">
                Autor Proiect: <span style="color: {PERIWINKLE}; font-weight: 700;">Camelia-Andreea Mărginean</span> | Grupa 1127 BDSA
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- CERINTA 1: INCARCARE SI FILTRARE ---
def cerinta_1():
    st.markdown('<h1 class="main-header">Cerinta 1: Incarcare si Filtrare</h1>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Alege fisierul CSV sau Excel:", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'): 
                st.session_state['df'] = pd.read_csv(uploaded_file)
            else: 
                st.session_state['df'] = pd.read_excel(uploaded_file)
            st.session_state['filename'] = uploaded_file.name
        except Exception as e: 
            st.error(f"Eroare la validarea fisierului: {e}")

    if st.session_state['df'] is not None:
        df = st.session_state['df']
        st.markdown(f"""
            <div class="success-box-green">
                S-au incarcat {df.shape[0]} randuri si {df.shape[1]} coloane din {st.session_state['filename']}.
            </div>
        """, unsafe_allow_html=True)
        st.write("### Previzualizare Date")
        st.dataframe(df.head(10), use_container_width=True)
        st.divider()
        
        df_filtered = df.copy()
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown(f'<p style="color:{PERIWINKLE}; font-weight:600; font-size:1.1rem;">Filtre Numerice</p>', unsafe_allow_html=True)
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                min_v, max_v = float(df[col].min()), float(df[col].max())
                if min_v < max_v:
                    is_integer = (df[col].dropna() % 1 == 0).all()
                    if is_integer:
                        val_range = st.slider(f"Gama {col}", int(min_v), int(max_v), (int(min_v), int(max_v)), step=1, key=f"s_{col}")
                    else:
                        val_range = st.slider(f"Gama {col}", min_v, max_v, (min_v, max_v), step=0.1, format="%.1f", key=f"s_{col}")
                    
                    # MODIFICARE AICI: Pastram valorile nule folosind | df_filtered[col].isna()
                    condition = ((df_filtered[col] >= val_range[0]) & (df_filtered[col] <= val_range[1])) | (df_filtered[col].isna())
                    df_filtered = df_filtered[condition]

        with col_f2:
            st.markdown(f'<p style="color:{PERIWINKLE}; font-weight:600; font-size:1.1rem;">Filtre Categorice</p>', unsafe_allow_html=True)
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                options = df[col].unique().tolist()
                # Pentru multiselect, tratam valorile nule daca exista in optiuni
                selected = st.multiselect(f"Selectează {col}", options, default=options, key=f"m_{col}")
                
                # MODIFICARE AICI: Pastram valorile nule si la categoriile care nu sunt selectate (sau raman in df)
                condition_cat = (df_filtered[col].isin(selected)) | (df_filtered[col].isna())
                df_filtered = df_filtered[condition_cat]

        st.markdown(f"""
            <div style="margin: 30px 0 20px 0; padding: 10px 5px;">
                <table style="width:100%; border:none; border-collapse: collapse;">
                    <tr style="border:none;">
                        <td style="width: 50%; text-align: center; border:none; padding: 0;">
                            <span style="color: #777; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px;">Eșantion Inițial</span><br>
                            <span style="color: {PERIWINKLE}; font-size: 2.8rem; font-weight: 300;">{len(df):,}</span>
                        </td>
                        <td style="width: 50%; text-align: center; border:none; padding: 0;">
                            <span style="color: #777; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px;">Rezultat Filtrare</span><br>
                            <span style="color: {PEACH}; font-size: 2.8rem; font-weight: 300;">{len(df_filtered):,}</span>
                        </td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_filtered, use_container_width=True)

# --- CERINTA 2 ---
def cerinta_2():
    st.markdown('<h1 class="main-header">Cerinta 2: Structura si Valori Lipsa</h1>', unsafe_allow_html=True)
    df = st.session_state['df']
    if df is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rânduri", len(df))
        m2.metric("Total Coloane", len(df.columns))
        mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
        m3.metric("Memorie", f"{mem_mb:.2f} MB")
        total_missing_perc = (df.isnull().sum().sum() / np.prod(df.shape) * 100)
        m4.metric("Valori Lipsă", f"{total_missing_perc:.1f}%")

        t1, t2, t3, t4 = st.tabs(["Preview", "Info", "Statistici", "Vizualizare"]) 

        with t1:
            st.write("### Primele Rânduri")
            num_rows = st.slider("Număr rânduri de afișat:", 1, 20, 5, key="preview_slider")
            st.dataframe(df.head(num_rows), use_container_width=True)
            with st.expander("Ultimele Rânduri"):
                st.dataframe(df.tail(num_rows), use_container_width=True)

        with t2:
            st.write("### Tipuri de Date")
            info_df = pd.DataFrame({
                'Coloana': df.columns,
                'Tip': [str(t) for t in df.dtypes],
                'Non-Null': df.notnull().sum().values,
                'Null': df.isnull().sum().values
            })
            
            c_info1, c_info2 = st.columns([2, 1])
            with c_info1:
                st.dataframe(info_df, use_container_width=True, hide_index=True)
            with c_info2:
                st.write("**Distribuția Tipurilor:**")
                type_counts = df.dtypes.astype(str).value_counts()
                fig_p = px.pie(values=type_counts.values, names=type_counts.index, 
                               hole=0.4, color_discrete_sequence=[PERIWINKLE, PEACH, "#ffd3b6"])
                st.plotly_chart(fig_p, use_container_width=True)

        with t3:
            st.write("### Statistici Descriptive")
            st.write("**Coloane Numerice:**")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.write("**Coloane Categorice:**")
            cat_df = df.select_dtypes(include=['object'])
            if not cat_df.empty:
                cat_stats = pd.DataFrame({
                    'Valori Unice': cat_df.nunique(),
                    'Cel Mai Comun': cat_df.mode().iloc[0],
                    'Frecvență': [cat_df[c].value_counts().iloc[0] for c in cat_df.columns],
                    'Procent': [(cat_df[c].value_counts().iloc[0] / len(df) * 100).round(2) for c in cat_df.columns]
                })
                st.dataframe(cat_stats, use_container_width=True)

        with t4:
            st.write("### Vizualizare Valori Lipsă")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                miss_df = pd.DataFrame({
                    'Coloană': missing.index, 
                    'Număr Lipsă': missing.values, 
                    'Procent': (missing.values/len(df)*100).round(2)
                })
                st.plotly_chart(px.bar(miss_df[miss_df['Număr Lipsă']>0], x='Coloană', y='Procent', 
                                       text='Număr Lipsă', title="Procentul Valorilor Lipsă pe Coloană",
                                       color_discrete_sequence=[PERIWINKLE]), use_container_width=True)
                
                st.dataframe(miss_df[miss_df['Număr Lipsă']>0], use_container_width=True, hide_index=True)
                
                st.write("### Heatmap Valori Lipsă (primele 50 rânduri)")
                st.caption("Galben = Lipsă, Albastru = Prezent")
                fig_h = px.imshow(df.head(50).isnull(), color_continuous_scale=[[0, PERIWINKLE], [1, PEACH]])
                fig_h.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.success("Dataset complet (fără valori lipsă)!")

# --- CERINTA 3: ANALIZA UNIVARIATA ---
def cerinta_3():
    st.markdown('<h1 class="main-header">Cerinta 3: Analiza Univariata</h1>', unsafe_allow_html=True)
    df = st.session_state['df']
    if df is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("Selectează coloana pentru analiză:", num_cols)
        n_bins = st.slider("Număr Bins pentru Histogramă:", 10, 100, 30)
        
        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            st.markdown(f"### Histogramă: {selected_col}")
            fig_hist = px.histogram(df, x=selected_col, nbins=n_bins, marginal="box", 
                                    title=f"Distribuție {selected_col}", color_discrete_sequence=[PERIWINKLE])
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_h2:
            st.markdown("### Statistici")
            s = df[selected_col]
            stats_data = pd.DataFrame({
                'Metrică': ['Minim', 'Q1 (25%)', 'Mediană', 'Q3 (75%)', 'Maxim', 'Media', 'Std Dev'],
                'Valoare': [s.min(), s.quantile(0.25), s.median(), s.quantile(0.75), s.max(), s.mean(), s.std()]
            }).round(4)
            st.dataframe(stats_data, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown(f"### Box Plot: {selected_col}")
        
        q1 = df[selected_col].quantile(0.25)
        q3 = df[selected_col].quantile(0.75)
        iqr = q3 - q1
        low_fence = q1 - 1.5 * iqr
        up_fence = q3 + 1.5 * iqr
        outliers = df[(df[selected_col] < low_fence) | (df[selected_col] > up_fence)]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Valori", len(df))
        m2.metric("Outlieri Găsiți", len(outliers))
        m3.metric("Procent Outlieri", f"{(len(outliers)/len(df)*100):.2f}%")

        fig_box = px.box(df, y=selected_col, points="outliers", color_discrete_sequence=[PEACH])
        fig_box.add_hline(y=up_fence, line_dash="dash", line_color="red", annotation_text="Upper Fence")
        fig_box.add_hline(y=low_fence, line_dash="dash", line_color="red", annotation_text="Lower Fence")
        st.plotly_chart(fig_box, use_container_width=True)

# --- CERINTA 4 ---
def cerinta_4():
    st.markdown('<h1 class="main-header">Cerinta 4: Analiza Categorica</h1>', unsafe_allow_html=True)
    df = st.session_state['df']
    if df is not None:
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            sel_cat = st.selectbox("Alege categoria:", cat_cols)
            counts = df[sel_cat].value_counts().reset_index()
            counts.columns = [sel_cat, 'Frecventa']
            counts['Procent %'] = (counts['Frecventa'] / len(df) * 100).round(2)
            
            st.plotly_chart(px.bar(counts, x=sel_cat, y='Frecventa', text='Frecventa', 
                                   color_discrete_sequence=[PERIWINKLE]), use_container_width=True)
            st.write("### Tabel Frecvențe")
            st.dataframe(counts, use_container_width=True)
        else: st.warning("Nu există coloane categorice.")

# --- CERINTA 5: CORELAȚIE ȘI OUTLIERI ---
def cerinta_5():
    st.markdown('<h1 class="main-header">Cerinta 5: Corelatie si Outlieri</h1>', unsafe_allow_html=True)
    df = st.session_state['df']
    if df is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.write("### Heatmap Corelație")
        st.plotly_chart(px.imshow(df[num_cols].corr(), text_auto='.2f', color_continuous_scale='Purples'), use_container_width=True)

        st.divider()
        st.write("### Scatter Plot (Analiză Bivariată)")
        col_s1, col_s2 = st.columns(2)
        v1 = col_s1.selectbox("Variabila X", num_cols, index=0, key="x5")
        v2 = col_s2.selectbox("Variabila Y", num_cols, index=min(1, len(num_cols)-1), key="y5")
        
        # SOLUTIE FINALA PENTRU DuplicateError:
        # Cream un DataFrame nou cu nume de coloane unice, indiferent daca v1 == v2
        df_plot = pd.DataFrame({
            'Axa_X': df[v1],
            'Axa_Y': df[v2]
        }).dropna()

        fig_scatter = px.scatter(
            df_plot, 
            x='Axa_X', 
            y='Axa_Y', 
            trendline="ols", 
            labels={'Axa_X': v1, 'Axa_Y': v2}, # Redenumim label-urile pe grafic sa arate frumos
            color_discrete_sequence=[PERIWINKLE], 
            title=f"Scatter: {v1} vs {v2}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Calcul Pearson
        corr_val, _ = pearsonr(df_plot['Axa_X'], df_plot['Axa_Y'])
        st.metric("Coeficient Pearson", f"{corr_val:.4f}")

        st.divider()
        st.write("### Detecție Outlieri (Metoda IQR)")
        
        out_res = []
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            low_limit = q1 - 1.5 * iqr
            up_limit = q3 + 1.5 * iqr
            
            n_out = len(df[(df[col] < low_limit) | (df[col] > up_limit)])
            perc_out = (n_out / len(df) * 100)
            
            out_res.append({
                "Coloană": col, 
                "Nr. Outlieri": n_out, 
                "Procent Outlieri (%)": round(perc_out, 2),
                "Limită Inferioară": round(low_limit, 2),
                "Limită Superioară": round(up_limit, 2)
            })
        
        st.dataframe(pd.DataFrame(out_res), use_container_width=True, hide_index=True)
        
        sel_box = st.selectbox("Vizualizare grafică outlieri:", num_cols, key="box_out")
        st.plotly_chart(px.box(df, y=sel_box, points="outliers", color_discrete_sequence=[PEACH]), use_container_width=True)
# --- MAIN ---
selected = sidebar_navigation()

if selected == "Descriere Aplicație":
    pagina_descriere()
elif selected == "Cerinta 1: Incarcare si Filtrare": 
    cerinta_1()
elif st.session_state['df'] is None: 
    st.info("Încarcă un fișier în Cerința 1 pentru a debloca restul paginilor.")
else:
    if selected == "Cerinta 2: Structura si Valori Lipsa": cerinta_2()
    elif selected == "Cerinta 3: Analiza Univariata": cerinta_3()
    elif selected == "Cerinta 4: Analiza Categorica": cerinta_4()
    elif selected == "Cerinta 5: Corelatie si Outlieri": cerinta_5()