## Deep Fire Topology: Understanding the role of landscape spatial patterns in wildfire occurrence using artificial intelligence
### Cristobal Pais, Alejandro Miranda, Jaime Carrasco, Zuo-Jun Max Shen

## Disclaimer
This software is for research use only. 

## Abstract
Increasing wildfire activity across the globe has become an urgent issue with enormous ecological and social impacts. 
While there is evidence that landscape topology affects fire growth, no study has yet reported its potential influence on fire ignition. 
This study proposes a deep learning framework focused on understanding the impact of different landscape topologies on the ignition of a 
wildfire and the rationale behind these results. Our model achieves an accuracy of above 90% in fire occurrence prediction, detection, and 
classification of risky areas by only exploiting topological pattern information from 17,579 landscapes. This study reveals the strong potential 
of landscape topology in wildfire occurrence prediction and its implications for similar research. The proposed methodology can be applied to 
multiple fields/studies to understand and capture the role and impact of different topological features and their interactions.

## Documentation (WIP)

## Requirements
- Python 3.6+
- numpy
- pandas
- tensorflow
- matplotlib
- seaborn
- tqdm
- opencv
- imread
- imutils

## Installation and Usage
Execute directly from the notebooks/scripts. 
Requires a Google Earth Engine account to use the data_generator_gee scripts.

## Output examples
### Negative observations sample
![Negative observations sample](data/sample_set/no_fire_collage_sampleSet.png)

### Positive observations sample
![Positive observations sample](data/sample_set/fire_collage_sampleSet.png)

### Conceptual network architecture and outputs 
End-to-end classification of an example (conceptual model). 
Land-cover input data is extracted from satellite images, being processed by the network. GradCAM, guided GradCAM, and filters visualizations are obtained at the different blocks of the network to improve the interpretability of the model.
![Conceptual network](conceptual_net.png)

### GradCAM collage (100 landscapes)
Land-cover images after applying the GradCAM method to highlight the most relevant areas when classifying the observation.
![GradCAM](exp_outputs/collages/gradCAM/gradCAM_collage.png)

### Zonal Statistics 
Filtered landscapes using different attention levels (full map, top 30\%, 50\%, and 75\% of the densest areas). Different land covers present in the area are highlighted in the landscape and filtered columns (see Landscape legend).
Red to blue colors represent a gradient from the most to the least relevant zones to classify an image as a positive case.
In this example, the network mainly highlights the impervious/urban land-cover, represented by areas covered by roads and cities.
![Zonal Statistics](exp_outputs/zonal_statistics/zonal_stats.png)

