Predicting house price data using Yelp/Zillow data
================

#### The goal of this project is to try to quantify what the role of textual data (in the form of restaraunt reviews) in predicting house price growth

To formalize this, we can think of the following model that states house
prices are a function of Yelp star ratings plus review data with
![\\epsilon\_{zy}](https://latex.codecogs.com/png.latex?%5Cepsilon_%7Bzy%7D
"\\epsilon_{zy}") capturing everything else the Yelp data doesn’t
capture. ![f()](https://latex.codecogs.com/png.latex?f%28%29 "f()")
represnts the model that uses both stars and rewview, and
![g()](https://latex.codecogs.com/png.latex?g%28%29 "g()") represents
the model that only uses stars.

  
![&#10;\\begin{aligned}&#10; P\_{zy}&=f(Star\_{zy},Review\_{zy}) +
\\epsilon\_{zy} \\\\&#10; P\_{zy}&=g(Star\_{zy}) +
\\epsilon\_{zy}&#10;\\end{aligned}&#10;](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Baligned%7D%0A%20%20%20%20P_%7Bzy%7D%26%3Df%28Star_%7Bzy%7D%2CReview_%7Bzy%7D%29%20%2B%20%5Cepsilon_%7Bzy%7D%20%5C%5C%0A%20%20%20%20P_%7Bzy%7D%26%3Dg%28Star_%7Bzy%7D%29%20%2B%20%5Cepsilon_%7Bzy%7D%0A%5Cend%7Baligned%7D%0A
"
\\begin{aligned}
    P_{zy}&=f(Star_{zy},Review_{zy}) + \\epsilon_{zy} \\\\
    P_{zy}&=g(Star_{zy}) + \\epsilon_{zy}
\\end{aligned}
")  
In terms of the variables: -
![P\_{zy}](https://latex.codecogs.com/png.latex?P_%7Bzy%7D "P_{zy}")-
House prices from `Zillow` -
![Star\_{zy}](https://latex.codecogs.com/png.latex?Star_%7Bzy%7D
"Star_{zy}")- Star ratings from `Yelp` -
![Review\_{zy}](https://latex.codecogs.com/png.latex?Review_%7Bzy%7D
"Review_{zy}")- Tex reviews from `Yelp`

and ![z](https://latex.codecogs.com/png.latex?z "z") is zipcode,
![y](https://latex.codecogs.com/png.latex?y "y") is the year

We can use machine learning models to predict ![\\hat{f}() \\ and \\
\\hat{g}()](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D%28%29%20%5C%20and%20%5C%20%5Chat%7Bg%7D%28%29
"\\hat{f}() \\ and \\ \\hat{g}()") (which the hat signifies). Once we
predict these functions, we can compare the predictions to see which one
is more accurate.

### Issue with levels vs difference

If our goal is to predict house price growth, the current specification
is insufficient because it will just tell us which zipcodes are
associated with high house prices. For example, in general, zipcodes
with higher restaurant ratings may have higher hosue prices. BUt that
doesn’t tell us much about how **changes** in restaurant ratings lead to
**changes** in house price growth. Therefore, we need to respecify the
equation in terms of differences as below. Furthermore, we want to look
at the change in **log** prices so that we can interpret these as
percentage changes.

  
![&#10;\\begin{aligned}&#10; \\Delta log(P\_{zy})&=f(\\Delta
Star\_{zy},\\Delta Review\_{zy}) + \\epsilon\_{zy} \\\\&#10; \\Delta
log(P\_{zy})&=g(\\Delta Star\_{zy}) +
\\epsilon\_{zy}&#10;\\end{aligned}&#10;](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Baligned%7D%0A%20%20%20%20%5CDelta%20log%28P_%7Bzy%7D%29%26%3Df%28%5CDelta%20Star_%7Bzy%7D%2C%5CDelta%20Review_%7Bzy%7D%29%20%2B%20%5Cepsilon_%7Bzy%7D%20%5C%5C%0A%20%20%20%20%5CDelta%20log%28P_%7Bzy%7D%29%26%3Dg%28%5CDelta%20Star_%7Bzy%7D%29%20%2B%20%5Cepsilon_%7Bzy%7D%0A%5Cend%7Baligned%7D%0A
"
\\begin{aligned}
    \\Delta log(P_{zy})&=f(\\Delta Star_{zy},\\Delta Review_{zy}) + \\epsilon_{zy} \\\\
    \\Delta log(P_{zy})&=g(\\Delta Star_{zy}) + \\epsilon_{zy}
\\end{aligned}
")  

Ongoing issues

  - The dataset we are provided is very unbalanced when it comes to
    geographical location
