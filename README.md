
## SIADS Capstone Project - Machine Learning Approaches to Dementia Biomarker Identification

Project Team 21: "The Neuromancers" - Casey Dye, Tony Lan, Camaron Mangham

### Highlights

- 32 genes of interest were selected from 50,281 genes via differential expression analysis of RNA-seq data
- A Support Vector Machine with a radial kernel achieved the highest recall and F1 score on a dementia prediction task
- Several genes stand out as important to model performance: KIAA0408, LOC728433, LOC100113421, CET4NP
- Two sets of the 32 genes were identified to be expressed similarly by hierarchical clustering between samples with dementia and without.
- Protein Analysis confirmed protein feature importance that was consistent with established and emerging literature

### Methods
- RNA-Seq differential expression analysis
- Bootstrapped model evaluation of differentially expressed genes
- Clustering analysis of differentially expressed genes
- Bootstrapped model evaluation of protein quantifications

### Context
Dementia is a condition with major impact across the globe. Dementia encompasses several conditions, with Alzheimer’s disease being the most prevalent form, that affect an individual's daily functioning with notable impact on memory and thinking. Over 55 million people worldwide live with one of these conditions; up to 24 million have Alzheimer’s disease specifically (Mayeux R, et al 2012). A majority of individuals with dementia live in low or middle income countries.

The impacts of dementia are realized in many ways. It is the seventh leading cause of death and a major cause of disability and dependency. Disproportionately affecting women, dementia causes higher disability-adjusted life years and mortality. Additionally, women provide a majority of care for those with dementia. The overall economic impact is estimated to be $1.3 Trillion (Dementia 2023). While there is no known cause of Alzheimer's Disease, the leading hypothesis is that it is caused by a combination of genetic and environmental factors  (Mayeux R, et al 2012). As a higher proportion of the population ages, understanding dementia becomes crucial to reduce the impact it has worldwide.

### Project Statement
Our aim is to further the understanding of dementia. Our current objective is to identify potential biomarkers using machine learning techniques that can be elaborated upon in future studies to combat dementia. To achieve this, we use a combination of bioinformatic and machine learning methods used in previous dementia studies to analyze genomic and proteomic data from the Aging, Dementia and Traumatic Brain Injury (TBI) Study.


### Dataset: Aging, Dementia and Traumatic Brain Injury (TBI) Study
The dataset we will be using for this project was developed by the Allen Institute for Brain Science in consortium with the University of Washington and Kaiser Permanente Washington Health Research Institute. These organizations undertook a longitudinal cohort-based study known as the Adult Changes in Thought (ACT) study (Aging, dementia and Traumatic Brain Injury Study, n.d.) . The data used in our analysis comes from a sample within this broader study. This particular group of participants had either experienced at least one traumatic brain injury with loss of consciousness or were part of the similarity-matched control group  (TECHNICAL WHITE PAPER: OVERVIEW 2017). For each participant, a post-mortem autopsy was performed that included dissection and banking of frozen brain tissue from fifteen regions. These tissues were used for immunohistochemistry, in situ hybridization, RNA sequencing, targeted proteomic analysis, quantification of free radical injury, gas chromatography-mass spectrometry, and immunoassays (TECHNICAL WHITE PAPER: QUANTITATIVE DATA GENERATION 2016).

### Data Access Statement
The dataset can be accessed [here](https://aging.brain-map.org/download/index). Some supplementary data is available from the API [here](http://api.brain-map.org).

### How to run this code:

Use deseq_gene_ml_requirements.txt for the following notebooks:
- DE_PyDeseq2.ipynb
- gene_ML_1.ipynb
- gene_ML_2.ipynb

Use TL_requirements.txt for the following notebooks:
- Protein_Differential_Expression_TL_FINAL.ipynb
- Protein_Gene_Exploration_TL_FINAL.ipynb
- Protein_Qualitative_Exploration_TL_FINAL.ipynb
- Protein_Quantitative_Exploration_Bootstrap_TL_FINAL.ipynb
- Protein_Quantitative_Exploration_Multicollinearity_TL_FINAL.ipynb
- Protein_Quantitative_Model_Tuning_TL_FINAL.ipynb