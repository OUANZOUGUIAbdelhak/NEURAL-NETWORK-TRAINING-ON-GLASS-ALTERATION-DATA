hey!!
you are an expert of writting articles in the journal npj materials degradation, and I want you to help me strucure my article based on what I will tell about my work I did. 
So lets start.
During my internship of one year, my principle sujbect was to create a data base on glass alteration from articles using open source generative ai tools and after that train a neural network on this data to predict the initial dissolution rate of glasses. 
I will expalin to you step by step so you can understand. 
here is the context: actually, the laboratory where I work, they are studing about glass, and the conservation of nuclear waste inside the glasses, in order to conserve that waste with a high activity for a long time and to controle that. but the glasses when they are in contact with an environment for instance water and under certain conditions, it provoqes the alteration of the glass by the time, and this alteration happens in differents steps or stages with differents speeds, the fisrt stage with the initiale dissolution rate, the second step with the residuel rate and finally with a possible alteration renewal, so the subject of my internship is to try to modelize the first speed ( the initial dissolution rate ). 
But the problem is that if we want to train a model to predict the initial dissolution rate of glass given some parameters we have to have some data first which wasn't the case. So we had first to create and extract the data that we can use to train the model and this using open source ai tools. 
so the first part of my internship was to collecte the data from articles and structure it in a data base and the biggest challenge here was to prouve that we can extract the data from articles automaticely using ai tools for example having a tool that can take an article in input and extract automaticly the data and stucture it in a table. 
so just to give you an idea of the structure of my data that I want to have at the end of my first part of my internship which is constructing a data base, here is a description of the structure of my data: very simply it will be a table with 107 columns each column represents a parameter and for rows it will be for each glass, so for exemple the first row  is the first glass that has 107 parameters, and the second row for the second glass with 107 parameter in the columns and keep going like this, so the purpose was to have the maximum of rows of glasses with all the parameters. and here is the parameters I'm talking about ( Type Document,Titre,Référence,Premier Auteur,Type de verre,Li,B,O,Na,Mg,Al,Si,P,K,Ca,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Cs,Ba,La,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,Ce,Pr,Nd,S_autres_TR,Th,U,Pu,Np,Am,Cm,S_autres_An,Somme,Densité,Homogénéité,B_IV,Irradié,Caractéristiques_si_irradié,Température,Statique_dynamique,Plage_granulométrique_si_poudre,Surface_spécifique_géométrique_si_poudre,Surface_spécifique_BET_si_poudre,Qualité_de_polissage_si_monolithe,Masse_du_verre,Surface_du_verre_S,Volume_de_la_solution_V,Débit_de_la_solution,pH_initial_T_amb,pH_initial_T_essai,Composition_de_la_solution,Durée_de_l_expérience,pH_final_T_amb,pH_final_T_essai,Normalisation_de_la_vitesse_Sgeo_ou_SBET,V0_Si_ou_r0_Si,r2_Si,Ordonnée_à_l_origine_Si,V0_B_ou_r0_B,Ordonnée_à_l_origine_B,V0_Na_ou_r0_Na,r2_Na,Ordonnée_à_l_origine_Na,V0_ΔM_ou_r0_ΔM,Congruence ) okey !! great !!
Now that we know and have an idea about the structure of the data base we want to build, I will explain to the different steps I came accros to acheive the first part of my internship.
Okey the first was to collect all the articles that are potentially pertinent to our data base, for example an article should have at least all the parameters that are mandatory for our data to predict the initiale dissolution rate. For instance the composition, pH, Temperature, and the initiale dissolution rate. But to say that an article is pertinent we have to find a very smart way instead of reading the article form A to Z, because even the abstract is not useful for us, because we can't jusdge if the article contians the relevent parameters from the abstract. so the first this we did is that we tryied to collect the maximum possible of articles that we found on the internet in the journals like  scienceDirect , google scholar, web of science and others and we trying just to relay on the titles of the articles to gather the maximum possible of this articles and there was also some articles that was given by my supervisors directly because they are experts of the domain. so after collecting as much as we can, and it was over 50 articles, and then we had to find a tool that can validate the pertinence of each article automaticly. so here I have to introduce the tool I use in my internship the most especially for this part which is the tool Lngflow, but I don't know how can I introduce it in my article, but well I rely on you to tell me what should I do based on what I will expain to you. so langflow is an open source ai tool that I hosted in my machine, I created a virtuel environment in my vscode version 1.96.2 and then I downloaded the library langflow  version 1.2.0 and then I open it in my local, and I had to create my first project which is question answer document, and the first thing I did is to create a prompt to define the criteria that the llm should use to build it's decission in whether an article is valid or not for the data base, so here is the prompt or as I like to call it the specification sheet ( and I hope I can conserve this name in the article too because it helps people understand well) : 
prompt { ### Instructions :

Lisez le document ci-dessous et déterminez si cet article est pertinent pour contribuer à une base de données visant à prédire la vitesse initiale d’altération des verres à l’aide de l’apprentissage automatique. Pour qu’un article soit considéré comme pertinent, il doit contenir au moins les informations suivantes pour chaque verre ou condition expérimentale évaluée :

**Document:**
{Document}


1. **Composition chimique du verre** :  
   - Fournie en pourcentage (sous forme d’oxydes ou d’éléments, par exemple : SiO2, B2O3, Na2O, etc.).  
   - La somme des pourcentages doit être proche de 100 % (tolérance ±5 %) pour représenter la quasi-totalité de la composition. Les compositions partielles ou incomplètes ne sont pas acceptables.

2. **Vitesse initiale d’altération explicite** :  
   - Fournie directement sous forme numérique avec une unité claire (ex. : g·m⁻²·d⁻¹, µm·d⁻¹, mg·m⁻²·h⁻¹).  
   - Peut être globale (pour le verre entier) ou spécifique à un élément (ex. : V₀,Si, V₀,B), mais doit être mesurée dans les conditions initiales de dissolution (avant saturation ou formation de couches altérées).  
   - Doit être normalisée (par ex., à la surface géométrique ou BET) avec la méthode de normalisation indiquée.  
   - Les données indirectes (ex. : courbes de concentration, épaisseurs de couche altérée) ou calculables (ex. : via des équations ou pentes) ne sont pas acceptables.

3. **Au moins un paramètre expérimental** parmi :  
   - pH initial (à température ambiante ou d’essai),  
   - Température de l’essai,  
   - Surface spécifique (BET ou géométrique, si poudre),  
   - Volume de la solution,  
   - Débit de la solution (pour essais dynamiques).  
   - Ce paramètre doit être clairement associé à l’expérience où la vitesse d’altération est mesurée.

Si l’un de ces trois critères (composition complète, vitesse explicite avec unité, et au moins un paramètre expérimental) est absent pour un verre ou une condition donnée, cet ensemble de données n’est pas pertinent. Évaluez chaque verre ou condition expérimentale séparément si l’article en présente plusieurs.

---

### Informations demandées :

1. **Pertinence de l’article** : [Oui/Non]  
2. **Justification** : [Explication concise indiquant si les critères sont remplis ou non, avec mention des éléments présents et absents pour chaque verre ou condition.]

---

### Format attendu :

1. **Pertinence de l’article** : [Oui/Non]  
2. **Justification** : [Texte expliquant pourquoi l’article est pertinent ou non, par exemple : "Pour le verre G0.40Nd4C, l’article fournit une composition complète (somme = 99,99 %), une vitesse initiale explicite (V₀,Si = 0,8 g·m⁻²·d⁻¹, normalisée à Sgeo) et la température (100 °C), donc pertinent." ou "La composition est donnée, mais aucune vitesse initiale explicite n’est fournie, seulement des courbes, donc non pertinent."]

---

### Instructions supplémentaires :

- **Données absentes** : Ne faites aucune supposition ; basez-vous uniquement sur les informations explicites du document.  
- **Vitesse initiale** : Acceptez uniquement les valeurs numériques explicites avec unité (ex. : "initial rate = 0,2 g·m⁻²·d⁻¹"), associées aux conditions initiales. Ignorez les courbes, équations, ou données indirectes (ex. : libération d’éléments sans vitesse calculée). Si l’unité manque ou si la normalisation n’est pas précisée, considérez la donnée invalide.  
- **Composition** : Vérifiez que la somme des pourcentages est proche de 100 % (tolérance ±5 %) ; sinon, rejetez la composition.  
- **Paramètres** : Assurez-vous qu’ils sont liés à l’expérience de la vitesse mesurée (ex. : température de l’essai, pas une température générale).  
- **Multiples ensembles** : Si l’article présente plusieurs verres ou conditions, évaluez chaque ensemble séparément et notez lesquels sont pertinents.  
- **Unités** : Les unités doivent être claires et convertibles en un standard (ex. : g·m⁻²·d⁻¹) si nécessaire.  
- **Qualité** : Si des incertitudes ou erreurs de mesure sont mentionnées pour la vitesse, notez-le comme bonus, mais cela n’affecte pas la pertinence.  
- **Focus** : Ignorez les sections non pertinentes (introduction, références) et concentrez-vous sur les résultats expérimentaux ou tableaux.

---

### Exemples :

- **Exemple positif** :  
  1. Pertinence : Oui  
  2. Justification : "Pour le verre G0.40Nd4C, l’article donne une composition complète (B 26,76 %, Si 33,09 %, etc., somme = 99,99 %), une vitesse initiale explicite (V₀,Si = 0,8 g·m⁻²·d⁻¹, normalisée à Sgeo), et la température (100 °C), donc pertinent."

- **Exemple négatif** :  
  1. Pertinence : Non  
  2. Justification : "L’article fournit la composition (somme = 99,99 %) et le pH initial (6), mais aucune vitesse initiale explicite, seulement des courbes de dissolution, donc non pertinent."

---
}

so after using this as a prompt and for the input of the llm I just said that we have {Répondre aux questions du prompt} and for the llm I used mistral small 22b which is hosted inside the company and I have access to it using an API, we analysed more than 50 documents using this tool and at the end we got 17 articles were valids for the data base and the tool rejected more than 33 articles, (I want to mention here that unfortonatly the articles we are seeking here is really rare that's why we ended up with that few number of articles). 
After that there was 3 articles that were given directly by my supervisors and they were valids but the problem is that they were a scanned document, so we had to find an open source tool that can be run loccaly to OCR them, so after searching and trying so many tools we ended up with two tools one of them is docling which is an open source tool that has some advantages and some disadvantages, the advantages is that it's rapid and easy but it's not precise 100% for the complexes scanned documents and also it doesn't keep the same format of the documents and we had to find a solution for that, and the solution was using the the machine printer in the laboratory that scans a document and by selecting the mode OCR with precision we end up with a great numeric docuement well converted 100% precise and keeps the same format of the original document. so at the end we used docling for 1 document and the internal machine printer for 2 documents and now we have 20 documents that are valides and ready to be analysed to extract the data from them. so after the two steps: collecting the articles, and preprocces some are documents that are scanned, and in total we have now 20 documents 11 of them are published by cea france and 3 from some laboratories from usa, and 2 from china and 2 from japan and 2 from france but from other resource different from cea, and average number of pages of this documents is 13.7 pages per article. Now we arrive to the 3 step which is building a tool that is capable of extracting automaticly the data from them and then send and structure this data in the table. this step is the most difficult step. So basicly we used the same tool langlfow to do this. but workflow we created is very long I really don't know how can I described it all and explain all the prompts and components and I don't know how to show all the resluts of it, but again I rely on you, I will try to give you the maximum of information of my workflow and you can tell me what to do. 
So the first thing in this step is that I have to tell you some of the main problems we ecountered during the phase we were trying to build this tool. So the first thing is that the document can have a big number of glasses per article, for exemple 25 glasses inside one article and for each glass we have to extract 107 parameters, which is very challenging for ai tools like open source llms to deal with regarding the limit of the number of tokens that can be generated by an llm. the second problem wich aggravate the first one is that in some articles we can find that they give multiple tests for each glass and for each test we have to extract 107 parameters. the third thing is that in diffrents articles the composition of glasses can be in differents form such as wt% oxide, wt% element, wt% cation, mol% oxide, mol% element, mol% cation so we have to make the tool able to extract all the types of forms and convert them to one unic form before sending them to the table. and also there was a problem of all the differents names that the parameters can have in differents articles for exemple the initiale dissolution rate can be represented as V0 in some articles but can be also r0 in some others,and also the units can be differents, so to solve all this problems we chose to use langflow and we created a workflow that is capable of taking an article as an input and extract all the glasses with there parameters, and then send them to an sql data base that receives the data and orginize it in a table. I will give you all the 

