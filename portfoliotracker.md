---
layout: page
title: PortfolioTracker
subtitle: Keep on top of your portfolio the easy way
---

PortfolioTracker is a free and open source portfolio tracking tool. It enables you to create your on database and manage your trades on multiple portfolios with ease. 

### How does it work?
Easy you first fork a copy of the [github repo](https://github.com/Aidzillafont/PortfolioTracker) and that will give you all the file you need to get started.  
In the repo you will find a file with the docker commands you need to start your mysql server.  
Once you have the server up you can then run the SQL scripts in the repo to create the database.
Finally load your trade files and price files using the PortfolioTracker python object and there your ready to **_track your portfolios the easy way_**

#### Database Structure

The database consists of various linked tables described in the below ER diagram.

Trades are loaded to the trades table and this creates new portfolios and assets if they don't exist. 
Prices are loaded to the prices table. From there you can start extracting reports using the python object PortfolioTracker

![ER Diagram](https://github.com/Aidzillafont/PortfolioTracker/blob/caccfa746857eb501060b1da32c25e17a500c778/PortfolioTracker.png)


