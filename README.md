# About:
This is a recreation of the DBSCAN algorithm using only scipy as a dependency. [Click here to read more about this project.](https://buildabuddha.github.io/2020-07-28-dbscan/)

This was done for the first 'build week' of the Computer Science portion of my Lambda School education. 

As a bonus, it includes an algorithm that... honestly, someone has probably done before, but I wanted to try making. I 
call it K-DBSCAN, and it works similarly to DBSCAN, except it accepts K (the number of clusters) as one of its inputs. 

K-DBSCAN will use a binary search pattern to attempt to locate an epsilon value that results in K 
clusters. This is helpful if you're not sure what epsilon value to use at first, and want a result that gives you a 
specific number of clusters. This is slower than a single DBSCAN, but this may save time fiddling with an epsilon 
value.   

# How to install:
```
!pip install -i https://test.pypi.org/simple/ DBSCAN-BuildABuddha
from DBSCAN.DBSCAN import KDBSCAN as KDBSCAN
from DBSCAN.DBSCAN import DBSCAN as DBSCAN
```
