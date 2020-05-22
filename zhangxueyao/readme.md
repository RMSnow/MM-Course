# Text Modal

## Statistics

### Train/Test Dataset

#### 划分标准

军事、政治类 -> Test

#### 正负样本

- Train

  ```
  1    16119
  0    16074
  Name: fake_label, dtype: int64
  
  1    0.500699
  0    0.499301
  Name: fake_label, dtype: float64
  ```

- Test

  ```
  0    891
  1    722
  Name: fake_label, dtype: int64
  
  0    0.552387
  1    0.447613
  Name: fake_label, dtype: float64
  ```

#### 类别

6种 + 2种（政治、军事）

- Train

  ```
  社会生活    20616
  医药健康     6320
  文体娱乐     2620
  财经商业     1455
  教育考试      901
  科技        281
  Name: category, dtype: int64
  
  社会生活    0.640388
  医药健康    0.196316
  文体娱乐    0.081384
  财经商业    0.045196
  教育考试    0.027987
  科技      0.008729
  Name: category, dtype: float64
  ```

- Test

  ```
  政治    1241
  军事     372
  Name: category, dtype: int64
  
  政治    0.769374
  军事    0.230626
  Name: category, dtype: float64
  ```

#### 是否有图片

- Train

  ```
  19006
  
  0.5903767899854006
  ```

- Test

  ```
  1061
  
  0.6577805331680099
  ```

## 词数统计

选择120

```
The longest sentence has 1631 words. When WORDS = 100, the cover_rate = 0.93
The longest sentence has 1631 words. When WORDS = 120, the cover_rate = 0.96
The longest sentence has 1631 words. When WORDS = 150, the cover_rate = 0.97
The longest sentence has 1631 words. When WORDS = 200, the cover_rate = 0.98
```