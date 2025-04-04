import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import requests
from scipy import stats

#dataframes
april = pd.read_csv('Youtube_data_1april.csv')
maart = pd.read_csv("youtube_data_5maart.csv")

#tabbar
image2 = Image.open('small_icon.png')
st.set_page_config(page_title='Youtube dashboard',
                   page_icon=image2,
                   layout="wide")

# Banner met CSS en afbeelding
st.markdown(
    """
    <style>
    .blue-banner {
        background-color: #706f6f;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .blue-banner h1 {
        color: black;
        margin: 0;
    }
    .blue-banner h3 {
        color: #000000);
        margin: 4px 0 0 0;
        font-weight: normal;
        font-size: 16px;
    </style>
    """,
    unsafe_allow_html=True
)

image = Image.open('youtube_Logo.png')

buffered = BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

st.markdown('''
    <div class="blue-banner">
        <div>
            <h1 style="color: #000000;">Youtube Dashboard</h1>
            <h3 style="color: #000000;">Gemaakt door Gem, Mariah en Roel</h3>
        </div>
        <div style="flex-shrink: 0;">
            <img src="data:image/png;base64,{}" style="width: 400px; height: 100px;" />
        </div>
    </div>
'''.format(img_str), unsafe_allow_html=True)

# Hoofdtabs
tabs = st.tabs(["Veranderingen 🔄😲", "Aanbevolen ⭐👍", "Kaart 🗺️📍","Maart 🌱🌦️", "April 🌸🌧️","Maart vs april ⚖️📊" ,"Voorspelling 🔮🧠"])

# Tab 1
with tabs[0]:
    st.title('Veranderingen 🔄😲')
    st.write("Op deze pagina worden de veranderingen en wijzigingen ten opzichte van het vorige dashboard weergegeven. Daarnaast zijn er veel nieuwe toevoegingen die het dashboard verrijken. Voor een compleet overzicht, bekijk het vernieuwde dashboard.")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image('Schermafbeelding 2025-04-03 190202.jpg', caption='Verbreding van dashboard', use_container_width=True)
    
    with col2:
        st.write("Als eerste hebben we het dashboard breder gemaakt, zodat de plots meer ruimte krijgen. Dit zorgt ervoor dat de visualisaties duidelijker en gemakkelijker te begrijpen zijn, wat de gebruikerservaring verbetert.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image('Schermafbeelding 2025-04-03 190718.jpg', caption='Tabs in plaats van een keuzemenu', use_container_width=True)

    with col2:
        st.write("Daarna hebben we het keuzemenu vervangen door tabs. Dit zorgt voor een professionelere uitstraling en neemt minder ruimte in beslag. Daarnaast hebben we een banner toegevoegd om het dashboard een visueel aantrekkelijke en samenhangende look te geven.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image('Schermafbeelding 2025-04-03 191020.jpg', caption='Top 10 vervangen voor top 3', use_container_width=True)

    with col2:
        st.write("Vervolgens hebben we de 'Top 10' vervangen door een 'Top 3'. Hierdoor is de lijst per land weggelaten. De informatie over de top 3 is nu op een visueel aantrekkelijke manier gepresenteerd, en gebruikers hebben de mogelijkheid om de video's direct af te spelen.")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.image('Schermafbeelding 2025-04-03 191432.jpg', caption='Top 5 verbeterd', use_container_width=True)

    with col2:
        st.write("De 'Top 5' categorieën zijn verbeterd. Ten eerste is er extra informatie toegevoegd over de top 3. Daarnaast is de slider vervangen door een dropdownmenu, wat het selecteren van een continent gemakkelijker maakt.")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.image('Schermafbeelding 2025-04-03 191632.jpg', caption='Verwijderd', use_container_width=True)

    with col2:
        st.write("De scatterplot van 'views vs. lengte van een video' is verwijderd. Daarnaast is zowel het volledige dataframe als het dataframe gefilterd op continent verwijderd.")        
#tab 2
with tabs[1]:
    # Continents and countries mapping
    continent_country_mapping = {
        'Africa': ['Algeria', 'Egypt', 'Ghana', 'Kenya', 'Libya', 'Morocco', 'Nigeria', 'Senegal',
                   'Tunisia', 'Tanzania', 'Uganda', 'South Africa', 'Zimbabwe'],
        'Asia': ['United Arab Emirates', 'Armenia', 'Azerbaijan', 'Bangladesh', 'Bahrain',
                 'Georgia', 'Indonesia', 'Israel', 'India', 'Iraq', 'Jordan', 'Japan', 'Cambodia',
                 'South Korea', 'Kuwait', 'Kazakhstan', 'Laos', 'Lebanon', 'Sri Lanka',
                 'Mongolia', 'Malaysia', 'Nepal', 'Oman', 'Philippines', 'Pakistan', 'Qatar',
                 'Saudi Arabia', 'Singapore', 'Thailand', 'Turkey', 'Taiwan', 'Vietnam', 'Yemen'],
        'Europe': ['Albania', 'Austria', 'Bosnia and Herzegovina', 'Belgium', 'Bulgaria',
                   'Belarus', 'Switzerland', 'Cyprus', 'Czechia', 'Germany', 'Denmark', 'Estonia',
                   'Spain', 'Finland', 'France', 'United Kingdom', 'Greece', 'Croatia', 'Hungary',
                   'Ireland', 'Iceland', 'Italy', 'Liechtenstein', 'Lithuania', 'Luxembourg',
                   'Latvia', 'Moldova', 'Montenegro', 'Malta', 'Netherlands', 'Norway',
                   'Poland', 'Portugal', 'Romania', 'Serbia', 'Russia', 'Sweden', 'Slovenia',
                   'Slovakia', 'Ukraine'],
        'North America': ['Costa Rica', 'Guatemala', 'Honduras', 'Mexico', 'Nicaragua', 'Panama',
                          'El Salvador', 'United States'],
        'Oceania': ['Australia', 'New Zealand', 'Papua New Guinea'],
        'South America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Peru',
                          'Paraguay', 'Uruguay', 'Venezuela']
    }

    # Dropdown for continents
    selected_continent = st.selectbox("Selecteer een continent:", list(continent_country_mapping.keys()))

    # Dropdown for countries based on selected continent
    if selected_continent:
        selected_region = st.selectbox("Selecteer een land:", continent_country_mapping[selected_continent])
        st.write(f"Je hebt gekozen voor {selected_region} in {selected_continent}.")
    
    # Functie om duur om te zetten naar seconden
    def duration_to_seconds(duration):
        try:
            parts = duration.split(":")
            parts = [int(p) for p in parts]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]
            elif len(parts) == 1:
                return parts[0]
            else:
                return 0
        except:
            return 0

    # Voeg short/long video toe
    april["video_type"] = april["duration"].apply(lambda x: "Short" if duration_to_seconds(x) < 60 else "Long")

    # Streamlit UI
    st.title("YouTube Video Recommender")

    # Overzicht van aantal video's per categorie en video type (Short/Long) voor het geselecteerde land
    country_filtered_df = april[april["land"] == selected_region]
    category_video_type_counts = country_filtered_df.groupby(["category_name", "video_type"]).size().unstack(fill_value=0)

    # Weergeven in Streamlit
    st.write(f"### Aantal video's per categorie en video type in {selected_region}:")
    st.bar_chart(category_video_type_counts)

    # Filters
    selected_video_type = st.radio("Short of lange video?", ["Short", "Long"])
    selected_category = st.selectbox("Selecteer een categorie:", april["category_name"].unique())

    # Filter dataset
    filtered_df = april[(april["land"] == selected_region) &
                        (april["video_type"] == selected_video_type) &
                        (april["category_name"] == selected_category)]

    # Weergeven in Streamlit
    st.write("### Aanbevolen video:")
    # Controleer of er video's zijn in de gefilterde dataset
    if not filtered_df.empty:
        # Controleer of de vereiste kolommen aanwezig zijn
        if "title" in filtered_df.columns and "video_id" in filtered_df.columns:
            # Als er nog geen video is geselecteerd, kies een standaard video (eerste video in de gefilterde dataset)
            if "selected_video" not in st.session_state:
                st.session_state.selected_video = filtered_df.iloc[0]

            # Knop om een andere video te tonen
            if st.button("Toon een andere video"):
                st.session_state.selected_video = filtered_df.sample(1).iloc[0]
                st.session_state["rerun_trigger"] = not st.session_state.get("rerun_trigger", False)  # Toggle a session state variable to trigger re-render

            # Toon de geselecteerde video
            st.markdown(f"*{st.session_state.selected_video['title']}*")
            st.video(f"https://www.youtube.com/watch?v={st.session_state.selected_video['video_id']}")
    else:
        st.write("Geen video's gevonden voor deze filters.")
    
#tab 3
with tabs[2]:
    # Krijg de top 5 video-titels
    Apriltop5 = april['title'].value_counts().head(5)
    top5_titles = Apriltop5.index.tolist()  # Haal de top 5 titels als een lijst

    # Als de sessie nog geen 'page' heeft, stel het in op de eerste video
    if 'page' not in st.session_state:
        st.session_state.page = top5_titles[0]

    # Functie om de kaart te maken
    def create_map(selected_video_title):
        # Filter de data voor de geselecteerde video
        dfEen = april[april['title'] == selected_video_title]

        # Haal de lijst van landen die in het dataframe van de specifieke video voorkomen
        landen_in_specifieke_video = dfEen['land'].unique()  # Lijst van landen voor deze specifieke video
        landen_in_april_df = april['land'].unique()  # Lijst van landen in de algemene April dataframe

        # Maak een Folium-kaart object
        m = folium.Map(location=[20, 0], zoom_start=2)

        # GeoJSON URL voor landen
        geojson_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"

        # Laad de GeoJSON data
        geojson_data = requests.get(geojson_url).json()

        # Voeg de landen toe met een kleur op basis van aanwezigheid in het dataframe
        for feature in geojson_data['features']:
            country_name = feature['properties']['name']

            # Haal de landnaam in het Nederlands uit de 'country' kolom
            land_nederlands = dfEen[dfEen['land'] == country_name]['country'].iloc[0] if country_name in landen_in_specifieke_video else country_name

            # Kleurinstelling op basis van aanwezigheid
            if country_name in landen_in_specifieke_video:
                kleur = 'green'  # Groen voor landen in de specifieke video
            elif country_name in landen_in_april_df:
                kleur = None  # Grijs voor landen die in April dataframe voorkomen maar niet in de specifieke video
            else:
                kleur = 'Gray'  # Geen kleur voor landen die nergens voorkomen

            # Voeg de landinformatie toe aan de kaart
            if kleur:
                folium.GeoJson(
                    feature,
                    style_function=lambda feature, color=kleur: {
                        'fillColor': color,
                        'color': 'black',  # Zwarte grenslijn
                        'weight': 1,  # Dunne lijn
                        'fillOpacity': 0.7
                    },
                    popup=folium.Popup(land_nederlands, max_width=200)  # Toon de naam in het Nederlands
                ).add_to(m)
            else:
                folium.GeoJson(
                    feature,
                    style_function=lambda feature: {
                        'color': 'black',  # Zwarte grenslijn voor landen zonder kleur
                        'weight': 1,  # Dunne lijn
                        'fillOpacity': 0  # Geen kleur voor landen die niet in het dataframe voorkomen
                    },
                    popup=folium.Popup(land_nederlands, max_width=200)  # Toon de naam in het Nederlands zonder kleur
                ).add_to(m)

        return m, landen_in_specifieke_video

    # Functie voor het renderen van de knoppen om video's te kiezen (met "Nummer x")
    def render_navigation():
        # Aantal knoppen
        num_buttons = len(top5_titles)

        # Dynamisch aantal kolommen aanmaken (knoppen naast elkaar plaatsen)
        cols = st.columns(num_buttons)  # Dynamisch aantal kolommen maken afhankelijk van het aantal knoppen

        # Maak de knoppen voor de top 5 video titels
        for i, title in enumerate(top5_titles):
            button_label = f" 🥇 Video {i+1}"  # Genereer een label als "Nummer 1", "Nummer 2", etc.
            
            with cols[i]:
                if st.button(button_label):
                    st.session_state.page = title  # Stel de geselecteerde video in op sessie

    st.title("**🗺 Waar worden de top 5 video's bekeken?**")

    # Render de navigatieknoppen
    render_navigation()

    # Maak de kaart en haal het aantal landen op
    map_for_selected_video, landen_in_specifieke_video = create_map(st.session_state.page)

    # Toon de titel van de pagina als "Nummer x"
    video_index = top5_titles.index(st.session_state.page)  # Vind het index van de geselecteerde video
    st.title(f" 🥇 Video {video_index + 1}")  # Toon de titel als "Nummer x"

    # Maak een kolomindeling voor de kaart en de informatie
    col1, col2 = st.columns([3, 1])  # De kaart krijgt 3 delen van de breedte, de informatie 1 deel

    # Voeg de kaart toe in kolom 1 met breder formaat (maximale breedte)
    with col1:
        st_folium(map_for_selected_video, width=1000, height=600)  # Verhoogde breedte van de kaart naar 1000 px

    # Voeg de informatie toe in kolom 2
    with col2:
        # Haal het kanaal en het aantal landen uit het dataframe
        dfEen = april[april['title'] == st.session_state.page]
        kanaal = dfEen['channel_title'].iloc[0]  # Nu gebruiken we de juiste kolomnaam
        aantal_landen = len(landen_in_specifieke_video)  # Aantal unieke landen waarin de video voorkomt

        # Toon de aanvullende informatie met emoji's voor een speelsere uitstraling
        st.subheader("🎬 Video Informatie:")
        st.markdown(f"**📽️ Titel:** {st.session_state.page}")
        st.markdown(f"**📺 Kanaal:** {kanaal}")
        st.markdown(f"**🌍 Aantal landen:** {aantal_landen}")

        # Voeg meer speelse info toe
        st.subheader("**Legenda:**")
        st.markdown("**🟢Groen**: In dit land treding")
        st.markdown("**⚪Wit**: In dit land niet trending")
        st.markdown("**⚫Grijs**: Geen data beschikbaar")


with tabs[5]:
    april2 = pd.read_csv("Youtube_data_1april.csv")
    st.metric(label="📈 Video's die in zowel maart als april trending waren", value=273)

    trending = pd.read_csv("trending.csv")

    # 📦 Unieke categorieën ophalen + "Alle categorieën" toevoegen
    categories = ["Alle categorieën"] + sorted(trending["category_name_april"].unique().tolist())

    # 🎛️ Filter via dropdown
    selected_category = st.selectbox("Selecteer een categorie:", options=categories)

    # 🧹 Filter de data
    if selected_category == "Alle categorieën":
        filtered_data = trending
    else:
        filtered_data = trending[trending["category_name_april"] == selected_category]

    # 📊 Histogram maken
    fig = px.histogram(
        filtered_data,
        x="rank_change",
        nbins=20,
        title=f"Verandering in rank (1 maand) - {selected_category}"
    )

    # ✏️ Styling toevoegen
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    fig.update_layout(
        xaxis_title="Verandering in rank",
        yaxis_title="Aantal",
        title=f"Verandering in rank (1 maand) - {selected_category}"
    )

    # ✅ Grafiek tonen in Streamlit
    st.subheader("📊 Histogram: verandering in trending-rank per categorie (1 maand)")
    st.plotly_chart(fig, use_container_width=True)

    # 📊 Data voorbereiden
    mean_maart = maart["duration_in_minutes"].mean()
    mean_april = april["duration_in_minutes"].mean()
    mean_trending = trending["duration_in_minutes_april"].mean()

    df = pd.DataFrame({
        "maand": ["maart", "april", "nog steeds trending"],
        "gemiddelde_duur": [mean_maart, mean_april, mean_trending]
    })

    # 📊 Balkgrafiek maken
    fig = px.bar(
        df,
        x="maand",
        y="gemiddelde_duur",
        text="gemiddelde_duur",
        title="Gemiddelde videoduur per maand"
    )

    # 🎨 Layout aanpassen
    fig.update_layout(
        xaxis_title="Periode",
        yaxis_title="Gemiddelde duur (minuten)"
    )

    # ✅ In Streamlit tonen
    st.subheader("⏱️ Gemiddelde videoduur per maand")
    st.plotly_chart(fig, use_container_width=True)

    # Aantal video's per categorie
    maart_cat = maart["category_name"].value_counts().reset_index()
    maart_cat.columns = ["category_name", "aantal"]
    maart_cat["maand"] = "maart"

    april_cat = april["category_name"].value_counts().reset_index()
    april_cat.columns = ["category_name", "aantal"]
    april_cat["maand"] = "april"

    df_cat_compare = pd.concat([maart_cat, april_cat])

    fig2 = px.bar(
        df_cat_compare,
        x="category_name",
        y="aantal",
        color="maand",
        barmode="group",
        title="📂 Vergelijking aantal video's per categorie (maart vs april)"
    )
    fig2.update_layout(
        xaxis_title="Categorie",
        yaxis_title="Aantal video's"
    )

    st.subheader("📊 Aantal video's per categorie (Maart vs April)")
    st.plotly_chart(fig2, use_container_width=True)

    # Top 3 maart
    top3_maart = (
        maart["category_name"]
        .value_counts()
        .head(3)
        .reset_index()
    )
    top3_maart.columns = ["category_name", "aantal"]
    top3_maart["maand"] = "maart"

    # Top 3 april
    top3_april = (
        april["category_name"]
        .value_counts()
        .head(3)
        .reset_index()
    )
    top3_april.columns = ["category_name", "aantal"]
    top3_april["maand"] = "april"

    # Combineer in één dataframe
    top3_combined = pd.concat([top3_maart, top3_april])


    fig = px.bar(
        top3_combined,
        x="category_name",
        y="aantal",
        color="maand",
        barmode="group",
        text="aantal",
        title="🏆 Vergelijking Top 3 categorieën – Maart vs April",
        labels={"category_name": "Categorie", "aantal": "Aantal video's"}
    )

    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)




with tabs[3]:
    def overview_maart():
        st.title("📈 Overzicht Maart")

        df = maart
        video_counts = df['video_id'].value_counts().reset_index()
        video_counts.columns = ['video_id', 'count']

        top_3_video_ids = video_counts.head(3)['video_id']
        top_3_videos = df[df['video_id'].isin(top_3_video_ids)]

        top_3_videos_info = top_3_videos[[
            'title', 'channel_title', 'video_id',
            'views', 'likes', 'comment_count',
            'published_at', 'category_name'
        ]].drop_duplicates(subset='video_id')

        top_3_videos_info = top_3_videos_info.merge(video_counts, on='video_id', how='left')
        top_3_videos_info = top_3_videos_info.sort_values(by='count', ascending=False).reset_index(drop=True)

        st.subheader("🏆 Top 3 Video’s in Maart")
        cols = st.columns(3)
        for i, row in top_3_videos_info.iterrows():
            with cols[i]:
                st.markdown(f"### Video {i+1}")
                st.video(f"https://www.youtube.com/watch?v={row['video_id']}")
                st.markdown(f"**🎬 Titel:** {row['title']}")
                st.markdown(f"**📺 Kanaal:** {row['channel_title']}")
                st.markdown(f"**📂 Categorie:** {row['category_name']}")
                st.markdown(f"**📅 Upload-datum:** {pd.to_datetime(row['published_at']).date()}")
                st.markdown(f"**👀 Views:** {int(row['views']):,}")
                st.markdown(f"**👍 Likes:** {int(row['likes']):,}")
                st.markdown(f"**💬 Comments:** {int(row['comment_count']):,}")
                st.markdown(f"**🔁 Voorkomen in dataset:** `{row['count']}` keer")

        st.subheader("📊 Interactieve scatterplot")
        y_axis_option = st.radio(
            "📈 Selecteer de Y-as:",
            options=["likes", "comment_count", "subscribers"],
            format_func=lambda x: {"likes": "Likes", "comment_count": "Comments", "subscribers": "Abonnees"}[x],
            index=0,
            key="y_axis_maart"
        )
        show_trendline = st.checkbox("Toon OLS Trendlijn", value=True, key="trendline_maart")

        youtube = df.copy()
        if "continent" in youtube.columns:
            continents = youtube["continent"].unique()
            youtube["continent_color"] = youtube["continent"]

        if youtube["views"].isna().sum() > 0 or youtube[y_axis_option].isna().sum() > 0:
            st.error("⚠ Er zitten ontbrekende waarden in de dataset.")
        else:
            scatter_fig = px.scatter(
                youtube,
                x="views",
                y=y_axis_option,
                color="continent",
                title=f"Views vs {y_axis_option.capitalize()} per Continent",
                log_x=True,
                log_y=True,
                trendline="ols" if show_trendline else None,
                hover_data={"title": True}
            )
            st.plotly_chart(scatter_fig)

        st.subheader("📊 Boxplot Videolengte per Continent")
        fig = px.box(
            youtube,
            x="continent",
            y="duration_in_minutes",
            color="continent"
        )
        fig.update_layout(
            yaxis=dict(type="log"),
            title="Boxplot Videolengte per Continent",
            xaxis_title="Continent",
            yaxis_title="Duur in minuten (log)"
        )
        st.plotly_chart(fig)

        # 📊 Top 3 categorieën bepalen
        st.write("")
        st.subheader("Top 3 Categorieën in Maart")
        top3_categorieën = (
            maart["category_name"]
            .value_counts()
            .head(3)
            .reset_index()
        )
        top3_categorieën.columns = ["category_name", "aantal"]

        # 🎛️ Toon als widgets in 3 kolommen
        cols = st.columns(3)

        for i, row in top3_categorieën.iterrows():
            with cols[i]:
                st.markdown(f"### Categorie {i+1}")
                st.markdown(f"**📂 Naam:** {row['category_name']}")
                st.markdown(f"**🔢 Aantal video's:** {int(row['aantal'])}")
        
        st.write("")
        st.write("")

        st.subheader("📊 Top 5 Categorieën per Continent")
        continent_options = sorted(df["continent"].unique().tolist())
        selected_continent = st.selectbox(
            "Selecteer een continent:", options=continent_options, key="continent_dropdown_maart"
        )

        continent_data = df[df["continent"] == selected_continent]
        top_categories = (
            continent_data["category_name"]
            .value_counts(normalize=True)
            .nlargest(5)
            .reset_index()
        )
        top_categories.columns = ["category_name", "percentage"]
        top_categories["percentage"] *= 100

        category_fig = px.bar(
            top_categories,
            x="category_name",
            y="percentage",
            color="category_name",
            title=f"Top 5 Categorieën in {selected_continent}"
        )
        st.plotly_chart(category_fig)

        # Ranking-analyse
        bins = list(range(1, 51, 5)) + [51]
        labels = [f"{i}-{i+4}" for i in range(1, 50, 5)]
        df['rank_group'] = pd.cut(df['rank_in_region'], bins=bins, labels=labels, right=False)

        grouped_counts = df.groupby(['rank_group', 'category_name']).size().unstack(fill_value=0).reset_index()
        st.subheader("📊 Ranking Analyse per Categorie")

        categories = df['category_name'].unique()
        selected_category = st.selectbox("Kies een categorie:", categories, key="categorie_ranking_maart")

        filtered_data = grouped_counts[['rank_group', selected_category]].rename(columns={selected_category: 'Aantal'})
        fig = px.bar(
            filtered_data,
            x='rank_group',
            y='Aantal',
            title=f'Aantal Voorkomen van Categorie {selected_category} per Ranking Groep',
            color='rank_group',
            text_auto=True
        )
        st.plotly_chart(fig)

        st.subheader("📊 Histogram: video's sinds upload per categorie (Maart)")

        # 📦 Categorieën uit maart
        categories_maart = maart["category_name"].unique()

        # 🎯 Beginnen met een lege figure
        fig_maart = go.Figure()

        # 🔁 Trace voor alle categorieën
        fig_maart.add_trace(go.Histogram(
            x=maart["days_since_upload"],
            nbinsx=20,
            name="Alle categorieën",
            marker_line_width=1,
            marker_line_color="black",
            visible=True  # standaard aan
        ))

        # 🔁 Eén trace per categorie (verborgen bij start)
        for cat in categories_maart:
            filtered = maart[maart["category_name"] == cat]
            fig_maart.add_trace(go.Histogram(
                x=filtered["days_since_upload"],
                nbinsx=20,
                name=cat,
                marker_line_width=1,
                marker_line_color="black",
                visible=False
            ))

        # 🔽 Dropdown bouwen
        dropdown_buttons_maart = [
            dict(
                label="Alle categorieën",
                method="update",
                args=[
                    {"visible": [True] + [False]*len(categories_maart)},
                    {"title": "Aantal video's in trendinglijsten sinds upload (Alle categorieën - Maart)"}
                ]
            )
        ]

        # 🔽 Eén knop per categorie
        for i, cat in enumerate(categories_maart):
            visible_list = [False] * (len(categories_maart) + 1)
            visible_list[i + 1] = True  # alleen deze categorie zichtbaar

            dropdown_buttons_maart.append(
                dict(
                    label=cat,
                    method="update",
                    args=[
                        {"visible": visible_list},
                        {"title": f"Aantal video's in trendinglijsten sinds upload ({cat} - Maart)"}
                    ]
                )
            )

        # 🔧 Layout
        fig_maart.update_layout(
            updatemenus=[
                dict(
                    buttons=dropdown_buttons_maart,
                    direction="down",
                    showactive=True,
                    x=1.05,
                    xanchor="left",
                    y=1,
                    yanchor="top"
                )
            ],
            xaxis_title="Dagen sinds upload",
            yaxis_title="Aantal video's",
            title="Aantal video's in trendinglijsten sinds upload (Alle categorieën - Maart)"
        )

        # ✅ Toon grafiek
        st.plotly_chart(fig_maart, use_container_width=True)


    overview_maart()


with tabs[4]:
    def overview_april():
        st.title("📈 Overzicht April")

        df = april
        video_counts = df['video_id'].value_counts().reset_index()
        video_counts.columns = ['video_id', 'count']

        top_3_video_ids = video_counts.head(3)['video_id']
        top_3_videos = df[df['video_id'].isin(top_3_video_ids)]

        top_3_videos_info = top_3_videos[[
            'title', 'channel_title', 'video_id',
            'views', 'likes', 'comment_count',
            'published_at', 'category_name'
        ]].drop_duplicates(subset='video_id')

        top_3_videos_info = top_3_videos_info.merge(video_counts, on='video_id', how='left')
        top_3_videos_info = top_3_videos_info.sort_values(by='count', ascending=False).reset_index(drop=True)

        st.subheader("🏆 Top 3 Video’s in April")
        cols = st.columns(3)
        for i, row in top_3_videos_info.iterrows():
            with cols[i]:
                st.markdown(f"### Video {i+1}")
                st.video(f"https://www.youtube.com/watch?v={row['video_id']}")
                st.markdown(f"**🎬 Titel:** {row['title']}")
                st.markdown(f"**📺 Kanaal:** {row['channel_title']}")
                st.markdown(f"**📂 Categorie:** {row['category_name']}")
                st.markdown(f"**📅 Upload-datum:** {pd.to_datetime(row['published_at']).date()}")
                st.markdown(f"**👀 Views:** {int(row['views']):,}")
                st.markdown(f"**👍 Likes:** {int(row['likes']):,}")
                st.markdown(f"**💬 Comments:** {int(row['comment_count']):,}")
                st.markdown(f"**🔁 Voorkomen in dataset:** `{row['count']}` keer")

        st.subheader("📊 Interactieve scatterplot")

        y_axis_option = st.radio(
            "📈 Selecteer de Y-as:",
            options=["likes", "comment_count", "subscribers"],
            format_func=lambda x: {"likes": "Likes", "comment_count": "Comments", "subscribers": "Abonnees"}[x],
            index=0,
            key="y_axis_april"
        )
        show_trendline = st.checkbox("Toon OLS Trendlijn", value=True, key="trendline_april")

        youtube = df.copy()

        # Kleur per continent (optioneel)
        if "continent" in youtube.columns:
            youtube["continent_color"] = youtube["continent"]

        # ➕ Nieuw: verwijder NaNs uit relevante kolommen
        youtube_filtered = youtube.dropna(subset=["views", y_axis_option])

        if youtube_filtered.empty:
            st.warning("⚠️ Geen data beschikbaar voor deze combinatie.")
        else:
            scatter_fig = px.scatter(
                youtube_filtered,
                x="views",
                y=y_axis_option,
                color="continent",
                title=f"Views vs {y_axis_option.capitalize()} per Continent",
                log_x=True,
                log_y=True,
                trendline="ols" if show_trendline else None,
                hover_data={"title": True}
            )
            st.plotly_chart(scatter_fig)


        st.subheader("📊 Boxplot Videolengte per Continent")
        fig = px.box(
            youtube,
            x="continent",
            y="duration_in_minutes",
            color="continent"
        )
        fig.update_layout(
            yaxis=dict(type="log"),
            title="Boxplot Videolengte per Continent",
            xaxis_title="Continent",
            yaxis_title="Duur in minuten (log)"
        )
        st.plotly_chart(fig)

        st.subheader("Top 3 Categorieën in April")
        top3_categorieën_april = (
        april["category_name"]
        .value_counts()            
        .head(3)
        .reset_index()
        )
        top3_categorieën_april.columns = ["category_name", "aantal"]

        cols = st.columns(3)
        for i, row in top3_categorieën_april.iterrows():
            with cols[i]:
                st.markdown(f"### Categorie {i+1}")
                st.markdown(f"**📂 Naam:** {row['category_name']}")
                st.markdown(f"**🔢 Aantal video's:** {int(row['aantal'])}")
        
        st.write("")
        st.write("")

        st.subheader("📊 Top 5 Categorieën per Continent")
        continent_options = sorted(df["continent"].unique().tolist())
        selected_continent = st.selectbox(
            "Selecteer een continent:", options=continent_options, key="continent_dropdown_april"
        )

        continent_data = df[df["continent"] == selected_continent]
        top_categories = (
            continent_data["category_name"]
            .value_counts(normalize=True)
            .nlargest(5)
            .reset_index()
        )
        top_categories.columns = ["category_name", "percentage"]
        top_categories["percentage"] *= 100

        category_fig = px.bar(
            top_categories,
            x="category_name",
            y="percentage",
            color="category_name",
            title=f"Top 5 Categorieën in {selected_continent}"
        )
        st.plotly_chart(category_fig)

        # Ranking-analyse
        bins = list(range(1, 51, 5)) + [51]
        labels = [f"{i}-{i+4}" for i in range(1, 50, 5)]
        df['rank_group'] = pd.cut(df['rank_in_region'], bins=bins, labels=labels, right=False)

        grouped_counts = df.groupby(['rank_group', 'category_name']).size().unstack(fill_value=0).reset_index()
        st.subheader("📊 Ranking Analyse per Categorie")

        categories = df['category_name'].unique()
        selected_category = st.selectbox("Kies een categorie:", categories, key="categorie_ranking_april")

        filtered_data = grouped_counts[['rank_group', selected_category]].rename(columns={selected_category: 'Aantal'})
        fig = px.bar(
            filtered_data,
            x='rank_group',
            y='Aantal',
            title=f'Aantal Voorkomen van Categorie {selected_category} per Ranking Groep',
            color='rank_group',
            text_auto=True
        )
        st.plotly_chart(fig)

        categories = april["category_name"].unique()

    # 🔁 Traces opbouwen (één per categorie + alles)
        fig = go.Figure()

    # Voeg trace toe voor "Alle categorieën"
        fig.add_trace(go.Histogram(
            x=april["days_since_upload"],
            nbinsx=20,
            name="Alle categorieën",
            marker_line_width=1,
            marker_line_color="black",
            visible=True  # standaard aan
        ))

        # Voeg per categorie een eigen trace toe
        for cat in categories:
            filtered = april2[april["category_name"] == cat]
            fig.add_trace(go.Histogram(
                x=filtered["days_since_upload"],
                nbinsx=20,
                name=cat,
                marker_line_width=1,
                marker_line_color="black",
                visible=False  # standaard uit
            ))

        # 🔽 Dropdown-knoppen bouwen
        dropdown_buttons = [
            dict(
                label="Alle categorieën",
                method="update",
                args=[
                    {"visible": [True] + [False]*len(categories)},
                    {"title": "Aantal video's in trendinglijsten sinds upload (Alle categorieën)"}
                ]
            )
        ]

        # Per categorie een knop toevoegen
        for i, cat in enumerate(categories):
            visible_list = [False] * (len(categories) + 1)  # +1 voor 'alles'-trace
            visible_list[i + 1] = True  # zet alleen de juiste categorie aan

            dropdown_buttons.append(
                dict(
                    label=cat,
                    method="update",
                    args=[
                        {"visible": visible_list},
                        {"title": f"Aantal video's in trendinglijsten sinds upload ({cat})"}
                    ]
                )
            )

        # 📦 Layout + dropdown
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    showactive=True,
                    x=1.05,
                    xanchor="left",
                    y=1,
                    yanchor="top"
                )
            ],
            xaxis_title="Dagen sinds upload",
            yaxis_title="Aantal video's",
            title="Aantal video's in trendinglijsten sinds upload (Alle categorieën)"
        )

        # ✅ In Streamlit tonen
        st.subheader("📊 Histogram: video's sinds upload per categorie April")
        st.plotly_chart(fig, use_container_width=True)

    overview_april()

with tabs[6]:
    st.title("Voorspelling 🔮🧠")
    st.write("Op deze pagina wordt een plot weergegeven die het aantal video's op het kanaal Alina Saito / 斎藤アリーナ toont. Daarnaast is er een voorspelling gedaan over het aantal video's dat aan het einde van 2026 op het kanaal te vinden zal zijn, met een verwachte hoeveelheid net onder de 2000 video's. Dit kanaal is gekozen omdat de meest trending video van april afkomstig is van dit kanaal, wat het extra interessant maakt voor de voorspelling.")
    # Geef het pad naar je bestand hier in
    video2 = pd.read_csv('video_details.csv')

    # Zorg ervoor dat je de juiste datetime-opmaak hebt
    video2["Published At"] = pd.to_datetime(video2["Published At"], format="%Y-%m-%dT%H:%M:%SZ", utc=True)
    video2["Published Date"] = video2["Published At"].dt.date

    # 📈 Tel aantal video's per publicatiedatum
    video_counts = video2.groupby("Published Date").size()

    # ✅ Bereken de cumulatieve som
    cumulative_counts = video_counts.cumsum()

    # ✅ Herkalibreer zodat de laatste waarde 1315 wordt
    adjustment = 1315 - cumulative_counts.iloc[-1]
    cumulative_counts += adjustment

    # 🔹 Zet de datums om naar numerieke waarden (dagen sinds eerste video)
    date_nums = np.array([(d - min(cumulative_counts.index)).days for d in cumulative_counts.index])

    # 🔹 Lineaire regressie om trend te berekenen
    slope, intercept, r_value, p_value, std_err = stats.linregress(date_nums, cumulative_counts)

    # 📅 *Begin van voorspelling: 4 april 2025, einde: 31 december 2026*
    start_prediction_date = pd.to_datetime("2025-04-04")
    future_dates = pd.date_range(start=start_prediction_date, end="2026-12-31")
    future_date_nums = np.array([(d.date() - min(cumulative_counts.index)).days for d in future_dates])
    predicted_videos = slope * future_date_nums + intercept

    # 🔹 Afronden van voorspelde waarden naar een heel getal
    predicted_videos_rounded = np.round(predicted_videos)

    # 🔹 Onzekerheid berekenen (95% betrouwbaarheidsinterval)
    confidence_interval = 1.96 * std_err * np.sqrt(1 + (1 / len(date_nums)) + ((future_date_nums - np.mean(date_nums)) ** 2 / np.sum((date_nums - np.mean(date_nums)) ** 2)))

    # Bereken de onder- en bovenlimiet van het betrouwbaarheidsinterval
    lower_bound = predicted_videos_rounded - confidence_interval
    upper_bound = predicted_videos_rounded + confidence_interval

    # 📊 Maak de interactieve Plotly-grafiek
    fig = go.Figure()

    # ✅ *Werkelijke data tot 3 april 2025, met blauwe lijn door de punten*
    fig.add_trace(go.Scatter(
        x=cumulative_counts.index, 
        y=cumulative_counts.values, 
        mode='lines+markers',  # 'lines+markers' om een lijn door de punten te trekken
        name='Cumulatieve Video\'s',
        line=dict(color='blue'),  # Lijnkleur blauw
        marker=dict(color='blue')  # Marker kleur blauw
    ))

    # ✅ *Voorspelde trend vanaf 4 april 2025 tot eind 2026, afgerond naar hele getallen*
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=predicted_videos_rounded, 
        mode='lines', 
        name='Voorspelde Trend (vanaf 4 april 2025)',
        line=dict(color='red', dash='dash')
    ))

    # ✅ *Onzekerheidsinterval vanaf 4 april 2025 tot eind 2026, als schaduw onder de lijn*
    fig.add_trace(go.Scatter(
        x=np.concatenate((future_dates, future_dates[::-1])),  # X-waarden tweemaal (boven+onder)
        y=np.concatenate((upper_bound, lower_bound[::-1])),  # Eerst boven, dan onder
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',  # Verhoog de transparantie van de schaduw
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Betrouwbaarheidsinterval'
    ))

    # ✅ Layout-instellingen
    fig.update_layout(
        title="Voorspelling Aantal Video's Tot Einde 2026",
        xaxis_title="Datum",
        yaxis_title="Cumulatief Aantal Video's",
        legend=dict(x=0, y=1),
        template="plotly_white"
    )

    # 📊 Toon de interactieve grafiek
    st.plotly_chart(fig)