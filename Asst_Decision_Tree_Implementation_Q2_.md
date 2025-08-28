# Decision Tree Classifier Performance from `classification-exp.py`

## A) 70/30 Train/Test

- **Accuracy**: 90.00%  
- **Macro Precision**: 90.00%  
- **Macro Recall**: 88.28%  

- **Class metrics**:
  - Class 0:
    - Precision: 0.9000
    - Recall: 0.9474
  - Class 1:
    - Precision: 0.9000
    - Recall: 0.8182

![Q2a Results](images\Figure2.png)

---

## B) Nested CV (5×5)

- **Best depth per outer fold**: [1, 4, 1, 1, 2]  
- **Outer-fold accuracies**: [0.8500, 0.9500, 0.9500, 0.8000, 0.9000]  
- **Mean outer accuracy**: 89.00%

![Q2b Results](images\Figure2(b).png)
