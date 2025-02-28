# Rosetta Docking & Candidate selection pipeline
 


## Overview
This repository provides a detailed pipeline for docking simulations of GnRH to GnRH1R. The process incorporates flexible docking and energy-based clustering to predict the most biologically relevant binding conformations.

## Steps

### 1. Inpute preparation (/Rosetta/Docking)
#### 1.1 Clean pdb
```bash
/path/to/rosetta/main/tools/protein_tools/scripts/clean_pdb.py 10A_start_AP.pdb AP
```
#### 1.2 Input Preparation
- Position GnRH and GnRH1R at a 10 Å distance.
- Predict the membrane environment using `mp_span_from_pdb`:
  ```bash
  /path/to/Rosetta/main/source/bin/mp_span_from_pdb.static.linuxgccrelease -in:file:s 10A_start_AP.pdb
  ```
#### 1.3 Minimisation of initial structure
run the minimisation.sh to minimise the initial structure and choose the lowest total score as input for the next stage

#### 1.4 Prepacking

run the prepack.sh and choose the lowest total score as input for the next stage

### 2. Docking
#### 2.1 Template Formation
- Generate 100 models and select the 15 best-scoring ones for the next step.
```bash
template.sh
```

### 2.2 Docking Protocol
- Run FlexPepDock.sh for further refinement of the selected templates.
 ```bash
FlexPepDock.sh
```
- Generate 15,000 docking poses  (1,0000 per template or adjust to your needs) and refine them further.
- Select the six best-scoring poses for the 2nd template and repeat the same process.
- Form 2,000 poses per pdb (=12,000 poses)
- This is the 2nd template (next clustering of this template) 

### 3. Energy-Based Clustering
- Cluster 12,000 structures using Rosetta energy-based clustering (use cluster.options):
  ```bash
  /path/to/Rosetta/main/source/bin/energy_based_clustering.static.li nuxgccrelease @cluster.options > cluster.log
  ```
- Select clusters based on Total Score and RMSD.

### 4. Interface Analysis
- Use Rosetta `InterfaceAnalyzer` to evaluate binding energies and buried surface areas:
  ```bash
  /path/to/Rosetta/main/source/bin/rosetta_scripts.static.linuxgccrele ase @Interface_Analyzer.options -in:file:s *c.*pdb > interfaceAnalyzer.log
  ```
- Use the Interface_Analyzer.options and Interface_Analyzer.xml files

### 5. Begining of candidate elimination (Rosetta/candidate_selection)

#### 5.1 Cluster Analysis
-Cluster selected for this analysis with population > 50
- Analyse cluster metrics (Total Score, ∆G binding, ∆SASA) using Python scripts:
  ```bash
  python total_clusters.py # 1st elimination stage 
  ```
- Only clusters 2, 4, 1, 5 were selected for further analysis

#### 5.2 Cluster statistical analysis

- Assesment of data distributions for the selection of statistical tests.
  ```bash
  python qq_plots.py
  ```
  -This script performs Kolmogorov-Smirnov (K-S) tests in conjunction with histograms and Q-Q plots for each metric. This offers insights into whether the data deviate from normal Gaussian distrinutions or not.

  -Violin plots (Mann-witney tests and effect size calculations of selected clusters)

```bash
python best_clusters_pValues_ES.py # 2nd elimination stage 
```

#### 5.3. Contact and Hydrogen Bond Analysis
- Generate contact heatmaps and H-bond analysis:
  ```bash
  python pdb_renumber.py #undo Rosetta residue numbering
  python find_contacts.py #3rd elimination stage
  python best_contacts.py #3rd elimination stage
  python contact_analysis.py #3rd elimination stage 
  python hbonds.py
  python GnRH_position.py
  
  ```

### 5.4. Binding Mode Selection for MD Simulations
- Rank candidate poses based on ∆G binding and key residue contacts:
  ```bash
  python Rank_contacts.py #4th elimination stage 
  python Candidate_selection.py #4th elimination stage 
  python Candidate_rank.py #4th elimination stage 
  ```
- Repack and minimise the selected candidates. Select the lowest total score for MD simulations


<table>
    <tr>
        <th>Elimination Criteria</th>
        <th>Stage 1 (Clustering results)</th>
        <th>Stage 2 (Cluster statistical analysis)</th>
        <th>Stage 3 (Cluster contact analysis)</th>
        <th>Stage 4 (Candidate selection)</th>
        <th>Selection</th>
    </tr>
    <tr>
        <td>Cluster population</td>
        <td>&lt; 50</td>
        <td>&lt; 1,000</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Total Score (REU)</td>
        <td>-</td>
        <td>&gt; -620</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ΔG Binding (REU)</td>
        <td>-</td>
        <td>&gt; 0</td>
        <td>-</td>
        <td>&gt; -20</td>
        <td>Lowest</td>
    </tr>
    <tr>
        <td>General contacts</td>
        <td>-</td>
        <td>-</td>
        <td>&lt; 1</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Important contacts</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>&lt; 1</td>
        <td>Highest</td>
    </tr>
</table>





