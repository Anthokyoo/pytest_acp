import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse d'Établissements Scolaires", layout="wide")

# Titre de l'application
st.title("🏫 Analyse en Composantes Principales (ACP)")
st.write("""
Cette application permet de réaliser une Analyse en Composantes Principales sur des données
concernant des établissements scolaires. L'ACP est une technique qui permet de réduire la
dimensionnalité des données tout en conservant le maximum d'information.
""")

st.info("💡 **Comment ça marche ?**\n1. Téléchargez votre fichier de données au format CSV.\n2. Sélectionnez les variables numériques que vous souhaitez inclure dans l'analyse.\n3. Choisissez éventuellement une variable catégorielle pour colorer les points sur le graphique.\n4. Explorez les résultats !")

# --- Section de téléchargement des données ---
st.header("1. Chargement des données")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';') # S'attendre à un séparateur point-virgule
        st.success("Fichier chargé avec succès ! Voici un aperçu des données :")
        st.dataframe(df.head())

        # --- Section de sélection des variables ---
        st.header("2. Sélection des variables pour l'ACP")
        
        # Filtrer pour ne garder que les colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.error("Aucune colonne numérique trouvée dans votre fichier. L'ACP ne peut être réalisée que sur des données numériques.")
        else:
            selected_cols = st.multiselect(
                "Choisissez les variables quantitatives pour l'analyse :",
                options=numeric_cols,
                default=numeric_cols[:min(len(numeric_cols), 5)] # Sélectionne les 5 premières par défaut
            )

            # Option pour la coloration des points
            categorical_cols = ['Aucune'] + df.select_dtypes(include=['object', 'category']).columns.tolist()
            color_option = st.selectbox(
                "Choisissez une variable qualitative pour la coloration du graphique :",
                options=categorical_cols
            )

            if st.button("🚀 Lancer l'Analyse ACP"):
                if len(selected_cols) < 2:
                    st.warning("Veuillez sélectionner au moins deux variables pour réaliser l'ACP.")
                else:
                    with st.spinner("Analyse en cours..."):
                        # Préparation des données
                        # On supprime les lignes avec des NaN uniquement sur les colonnes choisies pour l'analyse
                        X = df[selected_cols].dropna() 
                        
                        # Standardisation des données (étape cruciale pour l'ACP)
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        # Calcul de l'ACP
                        pca = PCA()
                        X_pca = pca.fit_transform(X_scaled)
                        
                        # Création d'un DataFrame avec les composantes principales
                        # Assurons-nous que le pca_df a le même index que X pour les jointures futures
                        pca_df = pd.DataFrame(
                            data=X_pca,
                            columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
                            index=X.index
                        )
                        
                        # --- Section des résultats ---
                        st.header("3. Résultats de l'Analyse")

                        # Éboulis des valeurs propres (variance expliquée)
                        st.subheader("📊 Variance expliquée par composante")
                        explained_variance = pca.explained_variance_ratio_ * 100
                        fig_scree = px.bar(
                            x=[f'PC{i+1}' for i in range(len(explained_variance))],
                            y=explained_variance,
                            labels={'x': 'Composantes Principales', 'y': 'Pourcentage de variance expliquée (%)'},
                            title="Éboulis des valeurs propres"
                        )
                        fig_scree.update_layout(yaxis_title="Pourcentage de variance (%)")
                        st.plotly_chart(fig_scree, use_container_width=True)
                        
                        # Projection des individus sur les deux premiers axes
                        st.subheader("📈 Projection des établissements sur les deux premiers axes")
                        
                        # Ajout de la variable de coloration si sélectionnée
                        if color_option != 'Aucune':
                            # On utilise .loc pour s'assurer de l'alignement des données
                            pca_df[color_option] = df.loc[X.index, color_option]
                        
                        # Ajout des noms des établissements pour le survol
                        # On suppose que la première colonne contient un identifiant unique
                        hover_name_col = df.columns[0]
                        pca_df['Etablissement'] = df.loc[X.index, hover_name_col]

                        fig_pca = px.scatter(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            color=color_option if color_option != 'Aucune' else None,
                            hover_name='Etablissement',
                            title=f"Projection sur PC1 ({explained_variance[0]:.2f}%) et PC2 ({explained_variance[1]:.2f}%)",
                            template='plotly_white'
                        )
                        fig_pca.update_layout(
                            xaxis_title=f"Axe Principal 1 ({explained_variance[0]:.2f}%)",
                            yaxis_title=f"Axe Principal 2 ({explained_variance[1]:.2f}%)",
                            legend_title_text=color_option
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)

                        # --- CERCLE DE CORRELATION ---
                        st.subheader("🌐 Cercle des corrélations")
                        
                        # Pour une ACP normée, les coordonnées des variables sont les composantes des vecteurs propres.
                        loadings = pca.components_.T
                        
                        fig_corr = go.Figure()

                        # Ajout des flèches et des noms pour chaque variable
                        for i, feature in enumerate(selected_cols):
                            fig_corr.add_trace(
                                go.Scatter(
                                    x=[0, loadings[i, 0]],
                                    y=[0, loadings[i, 1]],
                                    mode='lines+text',
                                    text=['', feature],
                                    textposition='top right',
                                    textfont=dict(size=12),
                                    line=dict(width=2),
                                    name=feature
                                )
                            )
                        
                        # Dessin du cercle unitaire
                        fig_corr.add_shape(
                            type='circle',
                            xref='x', yref='y',
                            x0=-1, y0=-1, x1=1, y1=1,
                            line_color='grey',
                            line_width=1.5,
                            layer='below'
                        )

                        # Mise en forme du graphique
                        fig_corr.update_layout(
                            title='Cercle des corrélations des variables sur les axes 1 et 2',
                            xaxis_title=f"Axe 1 ({explained_variance[0]:.2f}%)",
                            yaxis_title=f"Axe 2 ({explained_variance[1]:.2f}%)",
                            xaxis=dict(range=[-1.1, 1.1]),
                            yaxis=dict(
                                range=[-1.1, 1.1],
                                scaleanchor="x",
                                scaleratio=1
                            ),
                            showlegend=False,
                            template='plotly_white',
                        )
                        
                        st.plotly_chart(fig_corr, use_container_width=True)

                        st.success("Analyse terminée !")

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement du fichier : {e}")
        st.warning("Veuillez vérifier que votre fichier est un CSV valide et que le séparateur est un point-virgule (;).")

else:
    st.markdown("---")
    st.write("En attente du chargement d'un fichier de données.")
