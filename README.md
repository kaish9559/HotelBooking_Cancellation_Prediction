# HotelBooking_Cancellation_Prediction
      Hotel booking Project
I started with importing required libraries like NumPy, Pandas, Matplotlib, Seaborn, missingno etc.
I read the dataset file.
I checked the dataset and found its features and corresponding datatype.
Data types:
•	Categorical - hotel, is_canceled, customer_type, is_repeated_guest, meal, country, market_segment, distribution_channel, reserved_room_type, assigned_room_type, deposit_type, agent, company, reservation_status,
•	Numerical - lead_time, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, previous_cancellations, booking_changes, previous_bookings_not_canceled, days_in_waiting_list, adr, required_car_parking_spaces, total_of_special_requests.
•	Ordinal - arrival_date_year, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month, reservation_status_dat

I did statistical calculations on the dataset to find the impact of individual features on cancellation using the describe command and found that the following columns previous_cancellations, previous_bookings_not_canceled, booking_changes days_in_waiting_list, required_car_parking_spaces, total_of_special_requests have only a maximum value. This shows that these features contribute to the decision of cancellation only in very few cases.


Assumptions about the impact of features:
High: hotel, lead-time, arrival_date_year, arrival_date_month, stays_in_weekend_nights, stays_in_week_nights, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, reserved_room_type, assigned_room_type, deposit_type, days_in_waiting_list, customer_type
Medium: children, babies, distribution_channel, booking_changes, adr
Low: arrival_date_week_number, arrival_date_day_of_month, country, meal, adults, market_segment, agent, company, required_car_parking_spaces, total_of_special_requests, reservation_status, reservation_status_date

Assumptions about cancellation:
1.	The type of hotel decides the cancelation rate with higher cancellations in city hotels as compared to resort hotels due to variety of facilities available in resort hotels.
2.	The earlier the booking made, higher the chances of cancellation.
3.	Customers who have bookings for longer durations have a lesser chance of cancelling their booking.
4.	As more children or babies are involved in the booking, higher chances of cancellation.
5.	Old guest (is_repeated_guest=1) is less likely to cancel the current booking.
6.	If there are high previous cancellations, the possibility of cancellation of the current booking is also high.
7.	If the room assigned is not the same as the reserved room type, a customer might cancel the booking.
8.	The higher the number of changes made to the booking, lesser is the chance of cancellation due to the investment of time in curating the booking as per one's requirement.
9.	Bookings that are refundable or for which deposits were not made at the time of booking stand a high chance of cancelation.
10.	If the number of days in the waiting list is significant, a customer might make some other booking due to the uncertainty of confirmation of the current booking.

Target variable: is_canceled

Data Preprocessing:
a.	Data Cleaning
I checked duplicates and missing values present in the dataset.
There were duplicates in the dataset so I removed them by using the command: df.drop_duplicates(inplace= False) 
Missing values:
1.	country
2.	agent
3.	company

I deleted the company column as it has 94% null values. For the other column, I replaced it with 0.
I also estimated the guest address from where they came. Maximum was from Portugal.

Data Imbalance: Data was slightly very less imbalanced.

In Machine Learning and Data Science, we often come across a term called Imbalanced Data Distribution, which generally happens when observations in one of the classes are much higher or lower than the other classes.

EXPLORATORY DATA ANALYSIS:
a.	UNIVARIATE ANALYSIS (Checking the validity of assumptions)
Percentage cancelation= 0.37041628277075134
b.	Correlation of cancellation with other features:
highest positive correlations: lead_time followed by previous_cancellations.
highest negative correlations: total_of_special_requests, required_car_parking_spaces.
c.	Cancellation Rate in Resort and City Hotel:
City hotels have a higher chance of cancellation than resort hotels.
Maximum cancelations occur if the booking is made 60-70 days before the check-in date.
715 bookings don't have both weekday or weekend nights which could be an error in the data as this is not possible in real-life scenarios. Therefore, these rows can be eliminated from the dataset.
d.	Cancellation effect of new and old guests:
As seen in the correlation table, the above graph bolsters the evidence that most customers are newcomers and they are less likely to cancel their current booking. Old guests are less likely to cancel the booking (14%). Assumption 5 holds true.
e.	Previous cancellation effect on booking cancellation:
Maximum customers have 0 previous cancellations. They are less likely to cancel the current booking. However, customers who have cancelled once earlier are more likely to cancel the current booking. This also matches with the positive correlation between previous_cancellations and is_cancelled and supports Assumption 6
f.	bookings that are non-refundable are cancelled.
g.	Maximum bookings occur in 2016 in the months of July and August.


Feature Engineering:
 I dropped columns that were not useful and created new columns for reservation status dates as year, month, and day.
Encoded categorical variables: hotel	meal	market_segment	distribution_channel	reserved_room_type	deposit_type	customer_type	year	month	day 
We did encoding so that we find better features that are affecting booking cancellation.
We normalized values to bring them into a common scale using the log of variables, making it easier to compare and analysed data. Normalization also helps to reduce the impact of outliers and improve the accuracy and stability of statistical models.


Model:
I used logistic regression, random forest, decision tree, and KNN model for booking cancellation predictions.
The given dataset is a supervised classification dataset. It holds booking information for a city hotel and a resort hotel with information such as How and when the booking was made, the length of passengers’ stay with the number of parking slots available, and the number of adults, children, and babies. The Logistic regression, K-Nearest Neighbour, Decision Tree, and Random Forest algorithms are used to handle this supervised classification model. Among these four machine learning algorithms, Random Forest and Decision trees perform well for accuracy.
I also used k-fold cross-validation techniques.
