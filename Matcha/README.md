# Matcha
## 1. General
Matcha is a manually annotated dataset for term extraction, covering 2 languages (English and Kazakh), and 2 domains (Blockchain and Material Science).


## 2. Abbreviations
**Languages:**
* "en" = English
* "kaz" = Kazakh
  
**Domains:**
* "block" = Blockchain
* "material_sci" = Material Science

## 3. Data Structure
The Matcha repository is organized to provide easy access to textual data and annotations in the domains of Blockchain Technology and Material Science. At the top level, it includes essential documentation files such as README.md and sources.txt. The repository features annotated text files and a list of unique terms extracted from the data. The repository also provides a mirrored structure in the Kazakh language.
 
```
Matcha
├── README.md  
├── sources.txt  
│  
├── en  
│   ├── block  
│   │   └── annotated  
│   │      ├── annotations  
│   │      │   └── unique_annotation_lists  
│   │      │       └── block_en_terms.csv  # Contains unique terms extracted from the text  
│   │      └── texts  
│   │          ├── block_en_01.txt  # Annotated text document  
│   │          ├── block_en_02.txt  
│   │          ├── ...  
│   │          └── block_en_06.txt  
│   │  
│   └── material_sci  (equivalent to "block")  
│    
│   
│  
└── kaz (equivalent to "en")  
```



## 4. Annotations 
In the blockchain domain, there are 6 documents with 668 total extracted terms for English and 6 documents with 511 total extracted terms for Kazakh language. In the materials science domain there are 7 documents with 375 terms extracted for the English language.
| Language | Domain            | Documents | Terms | Words | Characters |
|----------|--------------------|-----------|-------|-------|------------|
| english  | Blockchain        | 6         | 994   | 18478 | 115774     |
| english  | Material Science  | 7         | 947   | 11569 | 81447      |
| kazakh   | Blockchain        | 6         | 692   | 14368 | 119515     |
| kazakh   | Material Science  | 7         | 363   | 9005  | 81758      |


