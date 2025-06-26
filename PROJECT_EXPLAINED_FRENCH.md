# Project Deep Dive — Copper Price Forecasting & Trading Strategy

## Overview

Ce projet a pour objectif de construire une stratégie de trading prédictive sur le cuivre, basée sur des modèles de machine learning appliqués à des indicateurs techniques. Il comprend la collecte des données, la feature engineering, le modèle Random Forest, le backtest, et un dashboard de visualisation.


## 🎯 Motivation et objectifs du projet

Le cuivre est un métal industriel clé, très sensible aux cycles économiques mondiaux, ce qui rend la prévision de son prix précieuse pour les traders et les analystes. Ce projet vise à :

- Collecter et analyser des données historiques sur le prix du cuivre  
- Concevoir des indicateurs techniques pertinents pour capter les signaux du marché  
- Entraîner un classificateur robuste (Random Forest) pour prédire la direction du prix  
- Backtester une stratégie de trading basée sur les signaux du modèle pour évaluer sa rentabilité  
- Fournir un tableau de bord interactif pour la visualisation et le suivi en temps réel  

Une stratégie prédictive sur le cuivre peut aider les acteurs du marché à anticiper les mouvements de prix, gérer les risques et identifier des opportunités de trading rentables sur ce métal volatil et économiquement important.

Ce projet est un exemple concret de **data science financière**, **feature engineering**, **apprentissage automatique** et **trading quantitatif**.


---
### 1. Pourquoi le cuivre ?

#### 1.1 Importance économique et industrielle

Le cuivre est un métal industriel **fondamental** largement utilisé dans les secteurs de l’énergie, la construction, l’électronique et les infrastructures. Sa demande reflète souvent la santé économique globale, car :

- Il est un **indicateur avancé** des cycles économiques mondiaux, notamment dans les économies émergentes et développées.
- La croissance des secteurs liés aux énergies renouvelables, aux véhicules électriques, et aux infrastructures urbaines augmente la demande en cuivre.

#### 1.2 Volatilité et opportunités de trading

- Le cuivre est **très liquide** sur les marchés des matières premières, avec des volumes d’échange importants.
- Sa **volatilité modérée à élevée** le rend intéressant pour les stratégies de trading à court terme.
- Les prix du cuivre sont sensibles à la géopolitique, aux politiques commerciales, et aux tensions d’offre/demande, offrant des signaux exploitables.

#### 1.3 Disponibilité des données

- Les données de prix sur le cuivre sont facilement accessibles et fiables via des sources comme Yahoo Finance (`yfinance`), ce qui facilite la mise en place d’un projet d’analyse quantitative.
- Le cuivre bénéficie d’une longue série historique qui permet de construire des modèles robustes.

#### 1.4 Couverture sectorielle diversifiée

- Travailler sur le cuivre permet de toucher à plusieurs thématiques (énergie, industrie, environnement), ce qui rend le projet à la fois pertinent pour le trading et intéressant d’un point de vue économique.

---

#### Cause à effet liant le cuivre au marché

Le cuivre, en tant que baromètre de l’activité économique mondiale, réagit rapidement aux changements macroéconomiques, aux politiques commerciales et aux tensions géopolitiques. Ainsi, une hausse de la demande industrielle ou un choc d’offre impacte directement son prix, ce qui influe à son tour sur les décisions des investisseurs et traders sur les marchés des matières premières. Cette dynamique crée un cercle de rétroaction entre la santé économique globale et le prix du cuivre, rendant ce métal crucial pour la prévision et la stratégie de trading.

#### Lien avec l’actualité récente

Par exemple, les tensions géopolitiques actuelles au Moyen-Orient, notamment liées à l’Iran, impactent les marchés des matières premières par des risques sur les chaînes d’approvisionnement et l’énergie. Ces perturbations peuvent indirectement affecter la demande industrielle en cuivre (via les coûts énergétiques et la confiance économique), entraînant une volatilité accrue des prix. [Voir analyse récente sur l’impact géopolitique sur les commodities](https://www.reuters.com/business/energy/trump-iran-oil-policy-impact-2025-06-23/).




## 2. Choix techniques et méthodologiques

### 2.1 Sélection des indicateurs techniques

- **MA20 & MA50** : capturent les tendances court et moyen terme, classiques en analyse technique.  
- **RSI (14 jours)** : mesure la dynamique et identifie les zones de surachat/survente.  
- **Bollinger Bands** : évaluent la volatilité et les potentiels retournements ou cassures.  
- **MACD** : indicateur de momentum suivi de tendance.  
- **Momentum & ROC** : quantifient la vitesse et amplitude des mouvements.  
- **Volatilité 20 jours** : mesure la dispersion des prix, utile pour la gestion du risque.

Ces indicateurs sont éprouvés et combinent des informations complémentaires : tendance, momentum et volatilité.

### 2.2 Pourquoi Random Forest ?

- **Robustesse** face au bruit des données financières.  
- **Capacité non-linéaire** à capturer des relations complexes sans nécessiter trop de réglages.  
- **Explicabilité** via l’importance des variables.  
- **Résistance au surapprentissage** grâce à l’agrégation d’arbres.  
- Parfait pour un projet exploratoire sans données massives.

### 2.3 Limites identifiées

- Pas encore pris en compte :  
  - Coûts de transaction et slippage  
  - Impact de la liquidité  
  - Risques macroéconomiques non modélisés  
  - Adaptation dynamique aux changements de régime de marché  
- Modèle statique entraîné sur données historiques — risque de perte de performance en cas de changement structurel.

---

## 3. Problèmes rencontrés et solutions

- **Nettoyage et qualité des données** : données manquantes ou aberrantes détectées puis imputées ou exclues.  
- **Feature engineering** : tests multiples d’indicateurs pour éviter redondance et multicolinéarité.  
- **Surapprentissage** : validation croisée et évaluation sur jeu test pour contrôler la généralisation.  
- **Déploiement du dashboard** : gestion de la compatibilité des versions des bibliothèques et performances via virtualenv.

---

## 4. Pistes d’amélioration futures

- Ajout d’une couche de **hyperparameter tuning** via GridSearch ou Bayesian Optimization.  
- Exploration de modèles plus avancés : XGBoost, LightGBM, réseaux neuronaux (LSTM, Transformers).  
- Intégration de données **macroéconomiques**, news sentiment, et fondamentaux (stocks, production).  
- Backtesting amélioré avec prise en compte des **frais de transaction, slippage et gestion du risque**.  
- Implémentation d’une stratégie adaptative ou apprentissage en ligne (online learning).  
- Déploiement sur **serveur cloud** (Heroku, AWS) avec accès sécurisé au dashboard.  
- Documentation et tests unitaires pour faciliter la maintenance.

---

## 5. Conclusion

Ce projet constitue une première étape solide vers des stratégies quantifiées sur le cuivre. Il illustre la combinaison efficace entre analyse technique, machine learning et backtesting. Avec les évolutions proposées, il peut devenir un outil puissant pour un trader assistant ou analyste quant.

---
