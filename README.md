# nCoV2019
## The dataset
  We will try to model the number of infected people in China.
  The dataset of number infected cases for this repository is taken from https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6 graph.
## The model
  The model we chose for this problem is SIS model. This choice has two main benefits:
  - People are not immunized after recovery from coronavirus, and so far 2019-nCoV has low death rate, thus Recovery factor can be considered 0
  - The analytical solution of SIS model has sigmoid form of function and is easy to get partial derivative.

![Image description](https://upload.wikimedia.org/wikipedia/commons/1/16/SIS.PNG)

The mathematical formula behind this model is:
Due to the conversion of population N, and everyday, there are $\gamma$I cases that either die or get cured. We can derive the following formula:

![formula 1](https://wikimedia.org/api/rest_v1/media/math/render/svg/0934138588adcfd8b863be3c4146a1f75eaddf66)

![formula 2](https://wikimedia.org/api/rest_v1/media/math/render/svg/4d4f9fc0563a23d87a12a95dbb48cd9fe74da056)

![formula 3](https://wikimedia.org/api/rest_v1/media/math/render/svg/d4007ba3ffec77a5a074486d70022ec848545106)

By processing the three formula, we can derive a formula for I alone, it is a first order ODE, which is quite easy to get the final analytic solution which is:

![formula 4](https://wikimedia.org/api/rest_v1/media/math/render/svg/2bbb7fa85bab0e6a73d51291f91825e9f37cce65)

where V, I_inf and I_0 are function of gamma and beta

## Learning gamma and beta

Because of sigmoid-like form of the final solution, we can easily derive the partial derivative of I with respect to gamma and beta. Detail of the formula can be found in the implementation in sis.py file.

By getting the derivative, we could use gradient descent to minimize the differences between actual data and the analytic solution, which is also a formula of gamma and beta. There are 270 learning epochs and the training sets only contain of days 22,23,24,25,27,30,32. 

However, we found it pretty fit with the current data, as the following result picture show.

First figure shows how fit the data is with the actual data 
![final 2](https://github.com/tson1997/nCoV2019/blob/master/corona/epc/89.jpg?raw=true)

Second figure shows the learning process (first subplot) and the final shape of the infectious function
![final](https://github.com/tson1997/nCoV2019/blob/master/corona/-0.0703_and_0.1007.jpg?raw=true)
