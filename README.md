# US Presidential Election 2016 Analysis
### Andrew Voorhees, Colton Stapper, Ruchita Patel

Analyzed and tried to figure out what made the outcome from 2012 election to 2016 elections different.
Sources from which the data was obtained:
* County level demographic data: https://github.com/benhamner/2016-us-election/tree/master/input/county_facts_saved
* County level election data: https://github.com/tonmcg/County_Level_Election_Results_12-16
* Census data: http://www.census.gov/popest/data/counties/asrh/2012/PEPSR5H.html

Using these datasets, we used demographic data for clustering algorithms as a way to predict which way a county were to vote in 2016.
Doing this, gave us a sense of what was different from 2012 to 2016. 

### Files of interest:
* Anaylsis.ipynb
  * K-means Cluster analysis on the demographic data
* DataExplore.ipynb
  * Analyzing clustering algorthim and exploratory analysis on differences of 2012 and 2016 elections.
* Exporation.ipynb
  * Principle component analysis on picking features that are useful to predicting final county.
* Project4_REPORT.pdf
  * Final report describing findings from all notebooks.
