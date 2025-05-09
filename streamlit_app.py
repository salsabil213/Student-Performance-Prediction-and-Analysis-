import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import joypy
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import numpy as np
from branca.colormap import linear
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
 
st.set_page_config(layout="wide")
 
# File path
FILE_PATH = "academic_performance_data.xlsx"
 
# City-university mapping
city_universities = {
    "BOGOTÁ D.C.": [
        'UNIVERSIDAD DE LOS ANDES',
        'UNIVERSIDAD NACIONAL ABIERTA Y A DISTANCIA UNAD',
        'UNIVERSIDAD ANTONIO NARIÑO', 'UNIVERSIDAD COOPERATIVA DE COLOMBIA',
        'POLITECNICO GRANCOLOMBIANO', 'UNIVERSIDAD NACIONAL DE COLOMBIA',
        'UNIVERSIDAD MILITAR"NUEVA GRANADA"',
        'CORPORACION UNIVERSITARIA MINUTO DE DIOS -UNIMINUTO',
        'FUNDACION UNIVERSIDAD DE AMERICA', 'UNIVERSIDAD SERGIO ARBOLEDA',
        'PONTIFICIA UNIVERSIDAD JAVERIANA', 'UNIVERSIDAD CATOLICA DE COLOMBIA',
        'FUNDACION UNIVERSITARIA LOS LIBERTADORES',
        'CORPORACION TECNOLOGICA INDUSTRIAL COLOMBIANA - TEINCO',
        'UNIVERSITARIA AGUSTINIANA- UNIAGUSTINIANA',
        'ESCUELA COLOMBIANA DE INGENIERIA"JULIO GARAVITO"',
        'UNIVERSIDAD SANTO TOMAS',
        'FUNDACION UNIVERSIDAD DE BOGOTA"JORGE TADEO LOZANO"',
        'UNIVERSIDAD MANUELA BELTRAN-UMB',
        'UNIVERSIDAD DISTRITAL"FRANCISCO JOSE DE CALDAS"', 'UNIVERSIDAD EL BOSQUE',
        'CORPORACION UNIFICADA NACIONAL DE EDUCACION SUPERIOR-CUN',
        'UNIVERSIDAD LA GRAN COLOMBIA', 'UNIVERSIDAD CENTRAL',
        'FUNDACION UNIVERSITARIA AGRARIA DE COLOMBIA -UNIAGRARIA',
        'UNIVERSIDAD ECCI', 'CORPORACION UNIVERSITARIA REPUBLICANA',
        'UNIVERSIDAD DE LA SALLE' ,'UNIVERSIDAD INCCA DE COLOMBIA',
        'CORPORACION UNIVERSITARIA  UNITEC', 'UNIVERSIDAD LIBRE',
        'CORPORACION UNIVERSIDAD PILOTO DE COLOMBIA',
        'FUNDACION UNIVERSIDAD AUTONOMA DE COLOMBIA -FUAC',
        'FUNDACION UNIVERSITARIA KONRAD LORENZ', 'FUNDACION UNIVERSITARIA CAFAM',
        'INSTITUCION UNIVERSITARIA DE COLOMBIA - UNIVERSITARIA DE COLOMBIA',
        'UNIVERSIDAD EAN' ,'ESCUELA TECNOLOGICA INSTITUTO TECNICO CENTRAL',
        'FUNDACION UNIVERSITARIA EMPRESARIAL DE LA CAMARA DE COMERCIO DE Bogotá',
        'ESCUELA DE INGENIEROS MILITARES'
    ],
    "MEDELLIN": [
        'UNIVERSIDAD EAFIT', 'UNIVERSIDAD DE ANTIOQUIA',
        'UNIVERSIDAD NACIONAL DE COLOMBIA',
        'UNIVERSIDAD AUTONOMA LATINOAMERICANA-UNAULA', 'UNIVERSIDAD DE MEDELLIN',
        'UNIVERSIDAD COOPERATIVA DE COLOMBIA',
        'UNIVERSIDAD PONTIFICIA BOLIVARIANA' ,'UNIVERSIDAD EIA',
        'UNIVERSIDAD DE SAN BUENAVENTURA',
        'POLITECNICO COLOMBIANO"JAIME ISAZA CADAVID"',
        'INSTITUCION  UNIVERSITARIA PASCUAL BRAVO' ,'COLEGIO MAYOR DE ANTIOQUIA',
        'INSTITUTO TECNOLOGICO METROPOLITANO'
    ],
    "CALI": [
            'UNIVERSIDAD AUTONOMA DE OCCIDENTE' ,'PONTIFICIA UNIVERSIDAD JAVERIANA',
            'UNIVERSIDAD DEL VALLE' ,'UNIVERSIDAD DE SAN BUENAVENTURA',
            'UNIVERSIDAD ICESI' ,'UNIVERSIDAD SANTIAGO DE CALI', 'UNIVERSIDAD LIBRE',
            'INSTITUCION UNIVERSITARIA ANTONIO JOSE CAMACHO - UNIAJC',
            'ESCUELA MILITAR DE AVIACION"MARCO FIDEL SUAREZ"',
    ],
    "BUCARAMANGA": [
        'UNIVERSIDAD DE SANTANDER - UDES' ,'UNIVERSIDAD INDUSTRIAL DE SANTANDER',
        'UNIVERSIDAD PONTIFICIA BOLIVARIANA',
        'CORPORACION UNIVERSITARIA DE INVESTIGACION Y DESARROLLO -"UDI"',
        'UNIVERSIDAD AUTONOMA DE BUCARAMANGA-UNAB' ,'UNIVERSIDAD SANTO TOMAS',
        'UNIVERSIDAD MANUELA BELTRAN-UMB' ,'UNIDADES TECNOLOGICAS DE SANTANDER',
    ],
    "BARRANQUILLA":[
        'UNIVERSIDAD DEL NORTE' ,'UNIVERSIDAD AUTONOMA DEL CARIBE',
        'CORPORACION UNIVERSIDAD DE LA COSTA, CUC', 'UNIVERSIDAD DEL ATLANTICO',
        'UNIVERSIDAD LIBRE', 'UNIVERSIDAD SIMON BOLIVAR',
        'CORPORACION UNIVERSITARIA AMERICANA',
        'CORPORACION POLITECNICO DE LA COSTA ATLANTICA'
    ],
    "CARTAGENA": [
        'FUNDACION UNIVERSITARIA TECNOLOGICO COMFENALCO - CARTAGENA',
        'UNIVERSIDAD TECNOLOGICA DE BOLIVAR',
        'FUNDACION UNIVERSITARIA COLOMBO INTERNACIONAL - UNICOLOMBO',
        'UNIVERSIDAD DE CARTAGENA',
        "UNIVERSIDAD DEL SINÚ 'Elías Bechara Zainúm' - UNISINÚ",
    ],
    "POPAYAN": [
        'UNIVERSIDAD DEL CAUCA' ,'CORPORACION UNIVERSITARIA AUTONOMA DEL CAUCA',
        'FUNDACION UNIVERSITARIA DE POPAYAN',
        'CORPORACION UNIVERSITARIA COMFACAUCA - UNICOMFACAUCA'
    ],
    "PASTO": [
        'INSTITUCION UNIVERSITARIA CENTRO DE ESTUDIOS SUPERIORES MARIA GORETTI',
        'UNIVERSIDAD DE NARIÑO',
        'CORPORACION UNIVERSITARIA AUTONOMA DE NARIÑO -AUNAR',
        'UNIVERSIDAD MARIANA'
    ],
    "ARMENIA": [
        'UNIVERSIDAD DEL QUINDIO',
        'ESCUELA DE ADMINISTRACION Y MERCADOTECNIA DEL QUINDIO',
        'CORPORACION UNIVERSITARIA EMPRESARIAL ALEXANDER VON HUMBOLDT - C.U.E.'
    ],
    "TUNJA": [
        'UNIVERSIDAD SANTO TOMAS' ,'UNIVERSIDAD DE BOYACA - UNIBOYACA',
        'UNIVERSIDAD PEDAGOGICA Y TECNOLOGICA DE COLOMBIA'
    ],
    "SINCELEJO": [
        'CORPORACION UNIVERSITARIA DEL CARIBE - CECAR' ,'UNIVERSIDAD DE SUCRE',
        'CORPORACION UNIVERSITARIA ANTONIO JOSE DE SUCRE - CORPOSUCRE'
    ],
    "PEREIRA": [
        'UNIVERSIDAD CATOLICA DE PEREIRA',
        'UNIVERSIDAD TECNOLOGICA DE PEREIRA - ITP', 'UNIVERSIDAD LIBRE'
    ],
    "MONTERIA": [
        'UNIVERSIDAD DEL SINÚ "Elías Bechara Zainúm" - UNISINÚ',
        'UNIVERSIDAD DE CORDOBA' ,'UNIVERSIDAD PONTIFICIA BOLIVARIANA',
    ],
    "NEIVA": [
        'CORPORACION UNIVERSITARIA DEL HUILA-CORHUILA',
        'UNIVERSIDAD SURCOLOMBIANA',
        'FUNDACION UNIVERSITARIA NAVARRA - UNINAVARRA'
    ],
    "MANIZALES": [
        'UNIVERSIDAD NACIONAL DE COLOMBIA' ,
        'UNIVERSIDAD AUTONOMA DE MANIZALES'
    ],
    "VILLAVICENCIO": [
        'CORPORACION UNIVERSITARIA DEL META',
        'UNIVERSIDAD DE LOS LLANOS'
    ],
    "CUCUTA": [
        'UNIVERSIDAD FRANCISCO DE PAULA SANTANDER' ,
        'UNIVERSIDAD LIBRE'
    ],
    "QUIBDO": [
       'UNIVERSIDAD TECNOLOGICA DEL CHOCO"DIEGO LUIS CORDOBA"',
        'FUNDACION UNIVERSITARIA CLARETIANA - UNICLARETIANA'
    ],
    "SANTA MARTA": ["UNIVERSIDAD DEL MAGDALENA - UNIMAGDALENA"],
    "VALLEDUPAR": ["UNIVERSIDAD POPULAR DEL CESAR"],
    "CALDAS": ["CORPORACION UNIVERSITARIA LASALLISTA"],
    "TULUA": ["UNIDAD CENTRAL DEL VALLE DEL CAUCA"],
    "SOLEDAD": ["INSTITUCION UNIVERSITARIA ITSA"],
    "SOGAMOSO": ["UNIVERSIDAD PEDAGOGICA Y TECNOLOGICA DE COLOMBIA"],
    "CHIA": ["UNIVERSIDAD DE LA SABANA"],
    "RIOHACHA": ["UNIVERSIDAD DE LA GUAJIRA"],
    "SAN GIL": ["FUNDACION UNIVERSITARIA DE SAN GIL - UNISANGIL"],
    "IBAGUE": ["UNIVERSIDAD DE IBAGUE"],
    "ENVIGADO": ["INSTITUCION UNIVERSITARIA DE ENVIGADO"],
    "ESPINAL": ["INSTITUTO TOLIMENSE DE FORMACION TECNICA PROFESIONAL"],
    "PAMPLONA": ["UNIVERSIDAD DE PAMPLONA"],
    "OCAÑA": ["UNIVERSIDAD FRANCISCO DE PAULA SANTANDER"],
    "FUSAGASUGA": ["UNIVERSIDAD DE CUNDINAMARCA-UDEC"],
    "GIRARDOT": ["CORPORACION UNIVERSIDAD PILOTO DE COLOMBIA"],
    "YOPAL": ["FUNDACION UNIVERSITARIA INTERNACIONAL DEL TROPICO AMERICANO"]
}
 
# Reverse mapping: University -> City
university_city_map = {
    university: city
    for city, universities in city_universities.items()
    for university in universities
}
 
# City coordinates
city_coords = {
    'BOGOTÁ D.C.': [4.7110, -74.0721],
    'MEDELLIN': [6.2442, -75.5812],
    'CALI': [3.4516, -76.5320],
    'BUCARAMANGA': [7.1193, -73.1227],
    'BARRANQUILLA': [10.9685, -74.7813],
    'CARTAGENA': [10.3910, -75.4794],
    'POPAYAN': [2.4448, -76.6147],
    'PASTO': [1.2136, -77.2811],
    'ARMENIA': [4.5339, -75.6811],
    'TUNJA': [5.5353, -73.3678],
    'SINCELEJO': [9.3047, -75.3978],
    'PEREIRA': [4.8087, -75.6906],
    'MONTERIA': [8.7479, -75.8814],
    'NEIVA': [2.9359, -75.2894],
    'MANIZALES': [5.0689, -75.5174],
    'VILLAVICENCIO': [4.1420, -73.6266],
    'CUCUTA': [7.8939, -72.5078],
    'QUIBDO': [5.6947, -76.6612],
    'SANTA MARTA': [11.2408, -74.1990],
    'VALLEDUPAR': [10.4631, -73.2532],
    'CALDAS': [6.0964, -75.6366],
    'TULUA': [4.0847, -76.1954],
    'SOLEDAD': [10.9177, -74.7675],
    'SOGAMOSO': [5.7140, -72.9331],
    'CHIA': [4.8619, -74.0329],
    'RIOHACHA': [11.5444, -72.9072],
    'SAN GIL': [6.5538, -73.1347],
    'IBAGUE': [4.4389, -75.2322],
    'ENVIGADO': [6.1759, -75.5918],
    'ESPINAL': [4.1522, -74.8834],
    'PAMPLONA': [7.3794, -72.6483],
    'OCAÑA': [8.2378, -73.3584],
    'FUSAGASUGA': [4.3456, -74.3668],
    'GIRARDOT': [4.3030, -74.7972],
    'YOPAL': [5.3378, -72.3959]
}
 
@st.cache_data
def load_data():
    df = pd.read_excel(FILE_PATH)
    # Add city column to dataframe based on university
    df['City'] = df['University'].map(university_city_map)
   
    # Print diagnostics about the data
    st.sidebar.write(f"Total records: {len(df)}")
    st.sidebar.write(f"Unique universities: {df['University'].nunique()}")
   
    return df
 
df = load_data()
 
# Title
st.title("Student Performance Analysis & Efficiency Prediction: Colombian Engineering Students")
st.markdown("We aimed to analyze and predict student efficiency among engineering students in Colombian universities. Our objective was to identify patterns in student performance and build a model to support early intervention for students at risk")
 
# Sidebar filters - City first, then university based on city selection
with st.sidebar:
    st.header("Filters")
    
    # First select city
    city = st.selectbox("Select City", ['All'] + sorted(df['City'].dropna().unique()))
    
    # Filter dataframe by city first
    if city != 'All':
        city_df = df[df['City'] == city]
        # Get universities for the selected city
        city_universities_list = sorted(city_df['University'].dropna().unique())
    else:
        city_df = df
        city_universities_list = sorted(df['University'].dropna().unique())
    
    # Then select university based on selected city
    university = st.selectbox("Select University", ['All'] + city_universities_list)
    
    # Apply final filter if university is selected
    if university != 'All':
        filtered_df = city_df[city_df['University'] == university]
    else:
        filtered_df = city_df
    
    # Update the main dataframe with the filtered one
    df_display = filtered_df.copy()
    
    # Display counts for filtered data
    st.write(f"Showing {len(df_display)} records")
    st.write(f"From {df_display['University'].nunique()} universities")
    
    # Add a reset button
    if st.button("Reset Filters"):
        city = 'All'
        university = 'All'
        filtered_df = df
        df_display = df.copy()

# ---------- Overview Map: Enhanced Student Distribution by University Performance ----------
st.subheader("Geographic Distribution of Universities by Performance")
 
@st.cache_data
def prepare_map_data():
    # Ensure we don't have duplicate university entries
    # Group by university and calculate mean percentile
    df_uni = df.groupby(['University']).agg(
        PERCENTILE=('PERCENTILE', 'mean'),
        Student_Count=('PERCENTILE', 'count'),
        City=('City', 'first')  # Take the first city for each university
    ).reset_index()
   
    # Ensure PERCENTILE is numeric and drop any NaNs
    df_uni['PERCENTILE'] = pd.to_numeric(df_uni['PERCENTILE'], errors='coerce')
    df_uni = df_uni.dropna(subset=['PERCENTILE'])
   
    # Add lat/lon based on city
    df_uni['latitude'] = df_uni['City'].map(lambda x: city_coords.get(x, [None, None])[0])
    df_uni['longitude'] = df_uni['City'].map(lambda x: city_coords.get(x, [None, None])[1])
   
    # Drop any rows with missing coordinates
    df_uni = df_uni.dropna(subset=['latitude', 'longitude'])
   
    return df_uni
 
df_uni = prepare_map_data()
 
 
# Initialize the map
m = folium.Map(location=[4.5709, -74.2973], zoom_start=6, tiles='CartoDB positron')
 
# Set up color scale
min_val = df_uni['PERCENTILE'].min()
max_val = df_uni['PERCENTILE'].max()
if min_val == max_val:
    # Avoid equal min/max causing error
    max_val += 0.1
colormap = linear.YlGnBu_09.scale(min_val, max_val)
 
# Create marker cluster for better visualization with many points
marker_cluster = MarkerCluster().add_to(m)
 
# Add markers to the cluster only (removed searchable layer)
for _, row in df_uni.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=8 + min(row['Student_Count'] * 0.02, 10),  # Size based on student count but capped
        color=colormap(row['PERCENTILE']),
        fill=True,
        fill_color=colormap(row['PERCENTILE']),
        fill_opacity=0.8,
        popup=folium.Popup(f"<b>{row['University']}</b><br>City: {row['City']}<br>PERCENTILE: {row['PERCENTILE']:.2f}<br>Students: {row['Student_Count']}", max_width=300),
    ).add_to(marker_cluster)
 
# Add color legend
colormap.caption = 'PERCENTILE Scale'
colormap.add_to(m)
 
# Display the map in Streamlit
st_folium(m, width=1000, height=600)

# Display filtered data
st.subheader("Filtered Data")
st.dataframe(df_display)

# ---------- Visualization 1: S11 vs PRO Histograms ----------
st.subheader("Saber 11 vs Saber PRO Score Distributions")
 
score_pairs = [
    ('Math_S11', 'Quantitative Reasoning_PRO'),
    ('Critical Reading_S11', 'Critical Reading_PRO'),
    ('Citizen Competencies_S11', 'Citizen Competencies_PRO'),
    ('English_S11', 'English_PRO')
]
 
fig1, axs = plt.subplots(2, 2, figsize=(16, 12))
 
for i, (s11_col, pro_col) in enumerate(score_pairs):
    ax = axs[i // 2, i % 2]
    
    # Use filtered data based on selection
    sns.histplot(df_display[s11_col], color='#014f86', kde=True, label=s11_col, stat="density", bins=30, alpha=0.6, ax=ax)
    sns.histplot(df_display[pro_col], color='#89c2d9', kde=True, label=pro_col, stat="density", bins=30, alpha=0.6, ax=ax)
    
    # Add selection info to title
    selection_info = ""
    if university != 'All':
        selection_info = f" - {university}"
    elif city != 'All':
        selection_info = f" - {city}"
    
    ax.set_title(f'{s11_col} vs. {pro_col}{selection_info}')
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.legend()
 
plt.tight_layout()
st.pyplot(fig1)
 
# ---------- Visualization 2: Gender-wise Subject Scores ----------
st.subheader("Average Scores by Gender Across Subjects")
 
score_columns = [
    'Math_S11', 'Critical Reading_S11', 'Citizen Competencies_S11', 'Biology_S11', 'English_S11',
    'Quantitative Reasoning_PRO', 'Critical Reading_PRO', 'Citizen Competencies_PRO',
    'English_PRO', 'Written Communication_PRO', 'Formulation of Engineering Projects_PRO'
]

# Use filtered data
gender_means = df_display.groupby('GENDER')[score_columns].mean().T.reset_index()
gender_means.rename(columns={'index': 'Subject'}, inplace=True)
melted_gender = gender_means.melt(id_vars='Subject', var_name='Gender', value_name='Average Score')
 
fig2, ax2 = plt.subplots(figsize=(16, 10))
sns.set_style("whitegrid")
custom_palette = ['#a9d6e5', '#013a63']
barplot = sns.barplot(data=melted_gender, x='Subject', y='Average Score', hue='Gender', palette=custom_palette, ax=ax2)
 
for p in barplot.patches:
    height = p.get_height()
    barplot.annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom', fontsize=9, color='black', rotation=0)
 
# Add selection info to title
selection_info = ""
if university != 'All':
    selection_info = f" - {university}"
elif city != 'All':
    selection_info = f" - {city}"
    
ax2.set_title(f'Average Scores by Gender Across Subjects{selection_info}', fontsize=16, weight='bold')
ax2.set_xlabel('Subject')
ax2.set_ylabel('Average Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig2)
 
# ---------- Visualization 3: Socioeconomic Level ----------
st.subheader("Average Scores by Socioeconomic Level")
 
# Use filtered data
socio_means = df_display.groupby('Socioeconomic Level')[score_columns].mean().T.reset_index()
socio_means.rename(columns={'index': 'Subject'}, inplace=True)
melted_socio = socio_means.melt(id_vars='Subject', var_name='Socioeconomic Level', value_name='Average Score')
 
fig3, ax3 = plt.subplots(figsize=(18, 10))
color_palette = ['#012a4a', '#014f86', '#468faf', '#a9d6e5']
sns.barplot(data=melted_socio, x='Subject', y='Average Score', hue='Socioeconomic Level', palette=color_palette, ax=ax3)
 
# Add selection info to title
selection_info = ""
if university != 'All':
    selection_info = f" - {university}"
elif city != 'All':
    selection_info = f" - {city}"
    
ax3.set_title(f'Average Scores by Socioeconomic Level{selection_info}')
ax3.set_xlabel('Subject')
ax3.set_ylabel('Average Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig3)
 
# ---------- Visualization 4: Ridge Plot for G_SC by Stratum ----------
st.subheader("Distribution of Global Score by Socioeconomic Stratum")
 
# Use filtered data
df_visualize = df_display.copy()

# Check if any data exists after filtering
if len(df_visualize) > 0:
    stratum_labels = {
        'Stratum 1': "Low Income",
        'Stratum 2': "Lower-Middle",
        'Stratum 3': "Middle",
        'Stratum 4': "Upper-Middle",
        'Stratum 5': "High Income",
        'Stratum 6': "Very High Income",
        'Unknown': "Unknown"
    }
    df_visualize['Stratum_Label'] = df_visualize['STRATUM'].map(stratum_labels)
    ordered_labels = ["Low Income", "Lower-Middle", "Middle", "Upper-Middle", "High Income", "Very High Income", "Unknown"]
    
    # Filter to include only strata that exist in the filtered data
    existing_strata = df_visualize['Stratum_Label'].unique()
    ordered_labels = [label for label in ordered_labels if label in existing_strata]
    
    if len(ordered_labels) > 0:  # Make sure we have at least one stratum
        df_visualize['Stratum_Label'] = pd.Categorical(df_visualize['Stratum_Label'], categories=ordered_labels, ordered=True)
        df_visualize = df_visualize.sort_values('Stratum_Label')
        
        color_palette = ['#012a4a', '#014f86', '#2a6f97', '#468faf', '#61a5c2', '#89c2d9', '#a9d6e5']
        # Adjust color palette to match the number of strata
        color_palette = color_palette[:len(ordered_labels)]
        
        fig4, axes = joypy.joyplot(
            df_visualize,
            by="Stratum_Label",
            column="G_SC",
            color=color_palette,
            linewidth=1.2,
            fade=True,
            figsize=(14, 8),
            legend=False,
            alpha=0.9
        )
        
        legend_patches = [Patch(color=color_palette[i], label=ordered_labels[i]) for i in range(len(ordered_labels))]
        plt.legend(
            handles=legend_patches,
            title='Socioeconomic Stratum',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=False
        )
        
        # Add selection info to title
        selection_info = ""
        if university != 'All':
            selection_info = f" - {university}"
        elif city != 'All':
            selection_info = f" - {city}"
            
        plt.title(f"Distribution of Global Scores by Socioeconomic Stratum{selection_info}", fontsize=16, weight='bold', pad=20)
        plt.xlabel("Global Score (G_SC)", fontsize=13)
        plt.tight_layout()
        st.pyplot(plt.gcf())
    else:
        st.warning("Not enough data to display the ridge plot after filtering.")
else:
    st.warning("No data available for the current selection.")


    #----------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------
# ----------------------------------
FEATURES = [
    'Math_S11','Critical Reading_S11','Citizen Competencies_S11',
    'Biology_S11','English_S11','Quantitative Reasoning_PRO',
    'Critical Reading_PRO','Citizen Competencies_PRO','English_PRO',
    'Written Communication_PRO','Formulation of Engineering Projects_PRO'
]

CLUSTER_EXPLANATIONS = {
    0: """
**High Achievers**  
- Likely college‑ready  
- Perform well on standardized evaluations  
- Often from well‑resourced schools or strong academic backgrounds
""",
    1: """
**Needs Support**  
- Potential but face structural or educational challenges  
- Possibly from under‑resourced schools  
- Could benefit from mentoring, foundational skill reinforcement, study strategies
"""
}

# ----------------------------------
@st.cache_resource
def perform_kmeans_clustering(df, features, n_clusters=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    return df, scaler, kmeans

@st.cache_resource
def train_efficiency_classifier(df, features, target_col='is_efficient'):
    X = df[features].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['saga'],
        'l1_ratio': [0.25, 0.5, 0.75],
        'class_weight': [None, 'balanced']
    }

    model = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    return scaler, best_model

# ----------------------------------
# User input section
st.markdown("#  Enter Your Scores")

user_vals = []

# SABER_11 Section
st.subheader("SABER 11 Scores")
saber11_feats = [
    'Math_S11','Critical Reading_S11','Citizen Competencies_S11',
    'Biology_S11','English_S11'
]
cols11 = st.columns(3)
for i, feat in enumerate(saber11_feats):
    label = feat.replace("_", " ")
    val = cols11[i % 3].number_input(label, 0.0, 100.0, 50.0, 1.0)
    user_vals.append(val)

# SABER_PRO Section
st.subheader("SABER PRO Scores")
saberpro_feats = [
    'Quantitative Reasoning_PRO','Critical Reading_PRO',
    'Citizen Competencies_PRO','English_PRO',
    'Written Communication_PRO','Formulation of Engineering Projects_PRO'
]
colspro = st.columns(3)
for i, feat in enumerate(saberpro_feats):
    label = feat.replace("_", " ")
    if feat == "Formulation of Engineering Projects_PRO":
        val = colspro[i % 3].number_input(label, 0.0, 300.0, 150.0, 1.0) / 3.0  # Normalize
    else:
        val = colspro[i % 3].number_input(label, 0.0, 100.0, 50.0, 1.0)
    user_vals.append(val)

# ----------------------------------
if st.button("📊 Get My Results"):
    # Load and preprocess data
    df_full = pd.read_csv("model_data.csv")
    df_full["Formulation of Engineering Projects_PRO"] = df_full["Formulation of Engineering Projects_PRO"] / 3.0

    # Clustering
    df_clustered, cluster_scaler, kmeans = perform_kmeans_clustering(df_full.copy(), FEATURES)

    # Identify high achievers cluster
    avg_scores = df_clustered.groupby('Cluster')[FEATURES].mean().mean(axis=1)
    high_achievers_label = avg_scores.idxmax()

    # Label efficiency
    df_clustered['average_score'] = df_clustered[[f for f in FEATURES if f != 'Formulation of Engineering Projects_PRO']].mean(axis=1)
    df_clustered['is_efficient'] = np.where(
        (df_clustered['Cluster'] == high_achievers_label) & (df_clustered['average_score'] > 60),
        1,
        0
    )

    # Train classifier
    clf_scaler, efficiency_model = train_efficiency_classifier(df_clustered, FEATURES, 'is_efficient')

    # Process user input
    user_input = np.array(user_vals).reshape(1, -1)
    user_scaled = cluster_scaler.transform(user_input)

    cluster_id = int(kmeans.predict(user_scaled)[0])
    display_cluster = 0 if cluster_id == high_achievers_label else 1

    st.markdown(f"## You belong to **Cluster {display_cluster}**")
    st.markdown(CLUSTER_EXPLANATIONS[display_cluster])

    # Efficiency classification
    eff_scaled = clf_scaler.transform(user_input)
    eff_pred = int(efficiency_model.predict(eff_scaled)[0])
    st.markdown(f"### Efficiency Prediction: **{'Efficient ✅' if eff_pred == 1 else 'Not Efficient ❌'}**")