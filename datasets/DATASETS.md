# Introduction to datasets

In this project we are aiming to use various datasets and sometimes a combination of them. A problem is that all datasets comes under different file/folder structure.

The idea is to create a single file/folder structure that could be interpreted by the machine learning algortihm. This file convention will be named **Data Source Convention**

## Data Source Convention

1. dataset is split into `/test` and `/train`
2. inside `/test` or `/train`, subfolders must represent label names
3. names and file extensiosn do not matter

Sample structure would look like
- `/datasets`
  - `/test`
    - `/covid`
    - `/normal`
    - `/pneumonia`
  - `/train`
    - `/covid`
    - `/normal`
    - `/pneumonia`

## Dataset sources

1. Chest X-ray (Covid-19 & Pneumonia) [(SOURCE)](https://www.kaggle.com/prashant268/chest-xray-covid19-pneumonia)
    - Total 6432 files (2.3 GB; 350KB file avg.)
    - 🥈 silver award
    - Labels
        - COVID19
        - NORMAL
        - PNEUMONIA
2. Chest X-Ray Images (Pneumonia) [(SOURCE)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
    - Total 5856 files (1.24GB; 211KB file avg.)
    - 🥇 gold award
    - Labels
        - NORMAL
        - PNEUMONIA
3. COVID-19 Radiography Database [(SOURCE)](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
    - Total 21200 files (776MB; 37KB file avg.)
    - 🥇 gold award
    - Labels
        - COVID
        - Lung_Opacity
        - Normal
        - Viral Pneumonia
4. Covid XRay Dataset [(SOURCE)](https://www.kaggle.com/ahemateja19bec1025/covid-xray-dataset)
    - Total 3092 files (1.14GB; 369KB file avg.)
    - 🥉 bronze award
    - Labels
        - 0 (NORMAL)
        - 1 (COVID)
5. **Dataset X - a combination of datasets number 1, 2 and 4**
    - Total 15380 files (4.68GB; 304KB file avg.)
    - Labels
        - COVID19
        - NORMAL
        - PNEUMONIA