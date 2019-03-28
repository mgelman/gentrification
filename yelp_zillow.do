*****************************************************************************
* Look at the relationship between the zillow house price index and yelp restaurant ratings within zipcode-year
******************************************************************************
* created: 03/26/2019 by Michael Gelman 
* modified: 03/26/2019 by Michael Gelman 

*change directory and load datset
	cd C:\Users\mgelman_admin\Documents\GitHub\yelp_dataset
	insheet using yelp_zillow.tsv, clear

*use log values
	gen l_value = log(value)
	tabstat l_value, by(stars)


*standard reg
	reg l_value stars

*with zipcode fixed effects
	areg l_value stars i.year, a(postal_code)


*collapse by zipcode year
	collapse value stars (count) count=l_value, by(year postal_code)
	

*want at least 100 obs per zipcode-year
	sum count, det
	keep if count>=50

*how many years of data? at least 5 years?
	egen years = count(stars), by(postal_code)
	keep if years >=5
	
*log and look at non-linear relationship
	gen l_value = log(value)
	tw lpolyci l_value stars 

*standard reg
	reg l_value stars
	reg l_value stars i.year

*with zipcode fixed effects
*effect goes away so is it just higher value places have higher stars?
	areg l_value stars i.year, a(postal_code)


*look at resid
	reg l_value i.year i.postal_code
	predict l_value_resid, resid

	reg stars i.year i.postal_code
	predict stars_resid, resid

	reg l_value_resid stars_resid
	tw lpolyci l_value_resid stars_resid

****************
*time series
*****************
	tsset postal_code year

	reg D.l_value D.stars 
	areg D.l_value D.stars , a(year)
	areg D.l_value L2D.stars LD.stars D.stars FD.stars, a(year)


	gen D_l=D.l_value
	gen D_stars=D.stars

	tw lpolyci D_l D_stars
