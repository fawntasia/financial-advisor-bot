"""
Database initialization script for Financial Advisor Bot.
Creates SQLite database with all tables and populates S&P 500 tickers.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATABASE_PATH = Path("data/financial_advisor.db")

# S&P 500 constituents as of 2024 (503 tickers)
SP500_TICKERS = [
    ("AAPL", "Apple Inc.", "Information Technology", "Technology Hardware, Storage & Peripherals", "1982-11-30"),
    ("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software", "1994-06-01"),
    ("NVDA", "NVIDIA Corporation", "Information Technology", "Semiconductors", "2001-11-30"),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "Broadline Retail", "2005-11-18"),
    ("GOOGL", "Alphabet Inc. Class A", "Communication Services", "Interactive Media & Services", "2014-04-03"),
    ("GOOG", "Alphabet Inc. Class C", "Communication Services", "Interactive Media & Services", "2014-04-03"),
    ("META", "Meta Platforms Inc.", "Communication Services", "Interactive Media & Services", "2013-12-23"),
    ("BRK.B", "Berkshire Hathaway Inc. Class B", "Financials", "Multi-Sector Holdings", "2010-02-16"),
    ("TSLA", "Tesla Inc.", "Consumer Discretionary", "Automobile Manufacturers", "2020-12-21"),
    ("LLY", "Eli Lilly and Company", "Health Care", "Pharmaceuticals", "1976-06-30"),
    ("V", "Visa Inc. Class A", "Financials", "Transaction & Payment Processing Services", "2009-12-21"),
    ("UNH", "UnitedHealth Group Incorporated", "Health Care", "Managed Health Care", "1994-07-01"),
    ("JPM", "JPMorgan Chase & Co.", "Financials", "Diversified Banks", "1975-06-30"),
    ("XOM", "Exxon Mobil Corporation", "Energy", "Integrated Oil & Gas", "1957-03-04"),
    ("MA", "Mastercard Inc. Class A", "Financials", "Transaction & Payment Processing Services", "2008-07-18"),
    ("JNJ", "Johnson & Johnson", "Health Care", "Pharmaceuticals", "1973-06-30"),
    ("AVGO", "Broadcom Inc.", "Information Technology", "Semiconductors", "2014-05-08"),
    ("HD", "The Home Depot Inc.", "Consumer Discretionary", "Home Improvement Retail", "1988-03-31"),
    ("PG", "Procter & Gamble Company", "Consumer Staples", "Personal Care Products", "1957-03-04"),
    ("CVX", "Chevron Corporation", "Energy", "Integrated Oil & Gas", "1957-03-04"),
    ("MRK", "Merck & Co. Inc.", "Health Care", "Pharmaceuticals", "1957-03-04"),
    ("ABBV", "AbbVie Inc.", "Health Care", "Pharmaceuticals", "2012-12-31"),
    ("COST", "Costco Wholesale Corporation", "Consumer Staples", "Hypermarkets & Super Centers", "1993-10-01"),
    ("PEP", "PepsiCo Inc.", "Consumer Staples", "Soft Drinks & Non-alcoholic Beverages", "1957-03-04"),
    ("KO", "The Coca-Cola Company", "Consumer Staples", "Soft Drinks & Non-alcoholic Beverages", "1957-03-04"),
    ("ADBE", "Adobe Inc.", "Information Technology", "Application Software", "1997-05-05"),
    ("WMT", "Walmart Inc.", "Consumer Staples", "Hypermarkets & Super Centers", "1957-03-04"),
    ("BAC", "Bank of America Corporation", "Financials", "Diversified Banks", "1976-06-30"),
    ("CRM", "Salesforce Inc.", "Information Technology", "Application Software", "2008-09-15"),
    ("TMO", "Thermo Fisher Scientific Inc.", "Health Care", "Life Sciences Tools & Services", "2003-08-29"),
    ("ACN", "Accenture plc", "Information Technology", "IT Consulting & Other Services", "2011-07-06"),
    ("MCD", "McDonald's Corporation", "Consumer Discretionary", "Restaurants", "1970-06-30"),
    ("LIN", "Linde plc", "Materials", "Industrial Gases", "1992-07-01"),
    ("NKE", "NIKE Inc. Class B", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods", "1988-11-01"),
    ("ABT", "Abbott Laboratories", "Health Care", "Health Care Equipment", "1964-03-31"),
    ("AMD", "Advanced Micro Devices Inc.", "Information Technology", "Semiconductors", "2017-03-20"),
    ("CMCSA", "Comcast Corporation Class A", "Communication Services", "Cable & Satellite", "2002-11-19"),
    ("DIS", "The Walt Disney Company", "Communication Services", "Movies & Entertainment", "1976-06-30"),
    ("VZ", "Verizon Communications Inc.", "Communication Services", "Integrated Telecommunication Services", "1983-11-30"),
    ("DHR", "Danaher Corporation", "Health Care", "Life Sciences Tools & Services", "1998-11-18"),
    ("INTU", "Intuit Inc.", "Information Technology", "Application Software", "2000-12-05"),
    ("INTC", "Intel Corporation", "Information Technology", "Semiconductors", "1976-06-30"),
    ("WFC", "Wells Fargo & Company", "Financials", "Diversified Banks", "1976-06-30"),
    ("TXN", "Texas Instruments Incorporated", "Information Technology", "Semiconductors", "2001-01-01"),
    ("CAT", "Caterpillar Inc.", "Industrials", "Construction Machinery & Heavy Trucks", "1957-03-04"),
    ("AMGN", "Amgen Inc.", "Health Care", "Biotechnology", "1992-01-02"),
    ("IBM", "International Business Machines Corporation", "Information Technology", "IT Consulting & Other Services", "1957-03-04"),
    ("PM", "Philip Morris International Inc.", "Consumer Staples", "Tobacco", "2008-03-31"),
    ("UNP", "Union Pacific Corporation", "Industrials", "Railroads", "1957-03-04"),
    ("PFE", "Pfizer Inc.", "Health Care", "Pharmaceuticals", "1957-03-04"),
    ("SPGI", "S&P Global Inc.", "Financials", "Financial Exchanges & Data", "1957-03-04"),
    ("C", "Citigroup Inc.", "Financials", "Diversified Banks", "1988-05-31"),
    ("RTX", "RTX Corporation", "Industrials", "Aerospace & Defense", "1957-03-04"),
    ("HON", "Honeywell International Inc.", "Industrials", "Industrial Conglomerates", "1957-03-04"),
    ("QCOM", "QUALCOMM Incorporated", "Information Technology", "Semiconductors", "1999-07-22"),
    ("BA", "The Boeing Company", "Industrials", "Aerospace & Defense", "1957-03-04"),
    ("GE", "GE Aerospace", "Industrials", "Aerospace & Defense", "1957-03-04"),
    ("AMAT", "Applied Materials Inc.", "Information Technology", "Semiconductor Equipment", "1995-03-16"),
    ("LOW", "Lowe's Companies Inc.", "Consumer Discretionary", "Home Improvement Retail", "1984-02-29"),
    ("NEE", "NextEra Energy Inc.", "Utilities", "Multi-Utilities", "1976-06-30"),
    ("UPS", "United Parcel Service Inc. Class B", "Industrials", "Air Freight & Logistics", "2002-07-22"),
    ("MS", "Morgan Stanley", "Financials", "Investment Banking & Brokerage", "1993-08-02"),
    ("GS", "The Goldman Sachs Group Inc.", "Financials", "Investment Banking & Brokerage", "2002-07-22"),
    ("BLK", "BlackRock Inc.", "Financials", "Asset Management & Custody Banks", "2011-04-04"),
    ("SBUX", "Starbucks Corporation", "Consumer Discretionary", "Restaurants", "2000-06-07"),
    ("T", "AT&T Inc.", "Communication Services", "Integrated Telecommunication Services", "1983-11-30"),
    ("MDT", "Medtronic plc", "Health Care", "Health Care Equipment", "1986-10-31"),
    ("LMT", "Lockheed Martin Corporation", "Industrials", "Aerospace & Defense", "1957-03-04"),
    ("DE", "Deere & Company", "Industrials", "Agricultural & Farm Machinery", "1957-03-04"),
    ("GILD", "Gilead Sciences Inc.", "Health Care", "Biotechnology", "2004-03-30"),
    ("AXP", "American Express Company", "Financials", "Consumer Finance", "1976-06-30"),
    ("CVS", "CVS Health Corporation", "Health Care", "Health Care Services", "1957-03-04"),
    ("SCHW", "The Charles Schwab Corporation", "Financials", "Investment Banking & Brokerage", "1997-06-02"),
    ("CI", "The Cigna Group", "Health Care", "Managed Health Care", "1976-06-30"),
    ("ELV", "Elevance Health Inc.", "Health Care", "Managed Health Care", "1976-06-30"),
    ("PLD", "Prologis Inc.", "Real Estate", "Industrial REITs", "2003-07-17"),
    ("MDLZ", "Mondelez International Inc.", "Consumer Staples", "Packaged Foods & Meats", "2012-10-02"),
    ("TJX", "The TJX Companies Inc.", "Consumer Discretionary", "Apparel Retail", "1985-09-30"),
    ("ADP", "Automatic Data Processing Inc.", "Industrials", "Human Resource & Employment Services", "1981-03-31"),
    ("SYK", "Stryker Corporation", "Health Care", "Health Care Equipment", "2000-12-12"),
    ("CB", "Chubb Limited", "Financials", "Property & Casualty Insurance", "2010-07-15"),
    ("MMC", "Marsh & McLennan Companies Inc.", "Financials", "Insurance Brokers", "1987-08-31"),
    ("TMUS", "T-Mobile US Inc.", "Communication Services", "Wireless Telecommunication Services", "2019-07-15"),
    ("COP", "ConocoPhillips", "Energy", "Oil & Gas Exploration & Production", "1957-03-04"),
    ("ZTS", "Zoetis Inc.", "Health Care", "Pharmaceuticals", "2013-06-21"),
    ("MO", "Altria Group Inc.", "Consumer Staples", "Tobacco", "1957-03-04"),
    ("SO", "The Southern Company", "Utilities", "Electric Utilities", "1957-03-04"),
    ("DUK", "Duke Energy Corporation", "Utilities", "Electric Utilities", "1976-06-30"),
    ("BDX", "Becton Dickinson and Company", "Health Care", "Health Care Equipment", "1972-09-30"),
    ("NFLX", "Netflix Inc.", "Communication Services", "Movies & Entertainment", "2010-12-20"),
    ("REGN", "Regeneron Pharmaceuticals Inc.", "Health Care", "Biotechnology", "2013-05-01"),
    ("BSX", "Boston Scientific Corporation", "Health Care", "Health Care Equipment", "1995-02-21"),
    ("EOG", "EOG Resources Inc.", "Energy", "Oil & Gas Exploration & Production", "2000-12-01"),
    ("SLB", "Schlumberger Limited", "Energy", "Oilfield Services", "1957-03-04"),
    ("MU", "Micron Technology Inc.", "Information Technology", "Semiconductors", "1994-09-27"),
    ("KDP", "Keurig Dr Pepper Inc.", "Consumer Staples", "Soft Drinks & Non-alcoholic Beverages", "2022-06-21"),
    ("AON", "Aon plc", "Financials", "Insurance Brokers", "1996-04-23"),
    ("ITW", "Illinois Tool Works Inc.", "Industrials", "Industrial Machinery", "1986-02-28"),
    ("EQIX", "Equinix Inc.", "Real Estate", "Data Center REITs", "2015-03-20"),
    ("PGR", "The Progressive Corporation", "Financials", "Property & Casualty Insurance", "1997-08-04"),
    ("CL", "Colgate-Palmolive Company", "Consumer Staples", "Personal Care Products", "1957-03-04"),
    ("WM", "Waste Management Inc.", "Industrials", "Environmental & Facilities Services", "2008-08-19"),
    ("CSX", "CSX Corporation", "Industrials", "Railroads", "1967-09-30"),
    ("FDX", "FedEx Corporation", "Industrials", "Air Freight & Logistics", "1980-12-31"),
    ("SHW", "The Sherwin-Williams Company", "Materials", "Specialty Chemicals", "1964-06-30"),
    ("HCA", "HCA Healthcare Inc.", "Health Care", "Health Care Facilities", "2015-01-27"),
    ("PYPL", "PayPal Holdings Inc.", "Financials", "Transaction & Payment Processing Services", "2015-07-20"),
    ("OXY", "Occidental Petroleum Corporation", "Energy", "Oil & Gas Exploration & Production", "1957-03-04"),
    ("ETN", "Eaton Corporation plc", "Industrials", "Electrical Components & Equipment", "1957-03-04"),
    ("EMR", "Emerson Electric Co.", "Industrials", "Electrical Components & Equipment", "1965-03-31"),
    ("MCO", "Moody's Corporation", "Financials", "Financial Exchanges & Data", "1998-07-01"),
    ("FCX", "Freeport-McMoRan Inc.", "Materials", "Copper", "2011-07-01"),
    ("PXD", "Pioneer Natural Resources Company", "Energy", "Oil & Gas Exploration & Production", "2008-12-22"),
    ("NSC", "Norfolk Southern Corporation", "Industrials", "Railroads", "1957-03-04"),
    ("APH", "Amphenol Corporation", "Information Technology", "Electronic Components", "2008-09-30"),
    ("APD", "Air Products and Chemicals Inc.", "Materials", "Industrial Gases", "1985-04-30"),
    ("CCI", "Crown Castle Inc.", "Real Estate", "Specialized REITs", "2012-03-14"),
    ("PSA", "Public Storage", "Real Estate", "Specialized REITs", "2005-08-19"),
    ("VRTX", "Vertex Pharmaceuticals Incorporated", "Health Care", "Biotechnology", "2013-09-23"),
    ("F", "Ford Motor Company", "Consumer Discretionary", "Automobile Manufacturers", "1957-03-04"),
    ("ECL", "Ecolab Inc.", "Materials", "Specialty Chemicals", "1989-01-31"),
    ("ICE", "Intercontinental Exchange Inc.", "Financials", "Financial Exchanges & Data", "2007-09-26"),
    ("GM", "General Motors Company", "Consumer Discretionary", "Automobile Manufacturers", "2013-06-06"),
    ("KMB", "Kimberly-Clark Corporation", "Consumer Staples", "Household Products", "1957-03-04"),
    ("MNST", "Monster Beverage Corporation", "Consumer Staples", "Soft Drinks & Non-alcoholic Beverages", "2012-06-28"),
    ("DG", "Dollar General Corporation", "Consumer Discretionary", "General Merchandise Stores", "2012-12-03"),
    ("SRE", "Sempra", "Utilities", "Multi-Utilities", "2017-03-17"),
    ("SNPS", "Synopsys Inc.", "Information Technology", "Application Software", "2017-03-16"),
    ("CDNS", "Cadence Design Systems Inc.", "Information Technology", "Application Software", "2017-09-18"),
    ("D", "Dominion Energy Inc.", "Utilities", "Multi-Utilities", "2016-11-30"),
    ("ORLY", "O'Reilly Automotive Inc.", "Consumer Discretionary", "Specialty Stores", "2011-03-31"),
    ("PSX", "Phillips 66", "Energy", "Refining & Marketing", "2012-05-01"),
    ("GIS", "General Mills Inc.", "Consumer Staples", "Packaged Foods & Meats", "1957-03-04"),
    ("AEP", "American Electric Power Company Inc.", "Utilities", "Electric Utilities", "1957-03-04"),
    ("DXCM", "Dexcom Inc.", "Health Care", "Health Care Equipment", "2020-05-12"),
    ("EXC", "Exelon Corporation", "Utilities", "Multi-Utilities", "1957-03-04"),
    ("CTVA", "Corteva Inc.", "Materials", "Fertilizers & Agricultural Chemicals", "2019-06-03"),
    ("KMI", "Kinder Morgan Inc.", "Energy", "Oil & Gas Storage & Transportation", "2012-05-25"),
    ("KR", "The Kroger Co.", "Consumer Staples", "Food Retail", "1957-03-04"),
    ("AIG", "American International Group Inc.", "Financials", "Multi-line Insurance", "1980-03-31"),
    ("HUM", "Humana Inc.", "Health Care", "Managed Health Care", "2012-12-10"),
    ("STZ", "Constellation Brands Inc. Class A", "Consumer Staples", "Distillers & Vintners", "2005-07-01"),
    ("MCK", "McKesson Corporation", "Health Care", "Health Care Distributors", "2021-06-21"),
    ("LRCX", "Lam Research Corporation", "Information Technology", "Semiconductor Equipment", "2012-06-19"),
    ("MAR", "Marriott International Inc. Class A", "Consumer Discretionary", "Hotels, Resorts & Cruise Lines", "1998-10-01"),
    ("WMB", "The Williams Companies Inc.", "Energy", "Oil & Gas Storage & Transportation", "1975-03-31"),
    ("MSI", "Motorola Solutions Inc.", "Information Technology", "Communications Equipment", "1957-03-04"),
    ("AJG", "Arthur J. Gallagher & Co.", "Financials", "Insurance Brokers", "2016-05-31"),
    ("TT", "Trane Technologies plc", "Industrials", "Building Products", "2010-11-17"),
    ("PH", "Parker-Hannifin Corporation", "Industrials", "Industrial Machinery", "1985-11-30"),
    ("NXPI", "NXP Semiconductors N.V.", "Information Technology", "Semiconductors", "2021-03-22"),
    ("PRU", "Prudential Financial Inc.", "Financials", "Life & Health Insurance", "2002-07-22"),
    ("O", "Realty Income Corporation", "Real Estate", "Retail REITs", "2015-04-07"),
    ("NEM", "Newmont Corporation", "Materials", "Gold", "1969-06-30"),
    ("HLT", "Hilton Worldwide Holdings Inc.", "Consumer Discretionary", "Hotels, Resorts & Cruise Lines", "2017-03-20"),
    ("BK", "The Bank of New York Mellon Corporation", "Financials", "Asset Management & Custody Banks", "1995-03-31"),
    ("JCI", "Johnson Controls International plc", "Industrials", "Building Products", "2010-08-27"),
    ("DOW", "Dow Inc.", "Materials", "Commodity Chemicals", "2019-04-01"),
    ("IDXX", "IDEXX Laboratories Inc.", "Health Care", "Health Care Equipment", "2017-01-05"),
    ("COF", "Capital One Financial Corporation", "Financials", "Consumer Finance", "1998-07-01"),
    ("IQV", "IQVIA Holdings Inc.", "Health Care", "Life Sciences Tools & Services", "2017-08-29"),
    ("SPG", "Simon Property Group Inc.", "Real Estate", "Retail REITs", "2002-06-26"),
    ("CNC", "Centene Corporation", "Health Care", "Managed Health Care", "2016-03-30"),
    ("MET", "MetLife Inc.", "Financials", "Life & Health Insurance", "2000-12-11"),
    ("CME", "CME Group Inc.", "Financials", "Financial Exchanges & Data", "2006-08-11"),
    ("DVN", "Devon Energy Corporation", "Energy", "Oil & Gas Exploration & Production", "2000-12-01"),
    ("ROST", "Ross Stores Inc.", "Consumer Discretionary", "Apparel Retail", "2009-12-21"),
    ("TGT", "Target Corporation", "Consumer Discretionary", "General Merchandise Stores", "1976-05-31"),
    ("SYY", "Sysco Corporation", "Consumer Staples", "Food Distributors", "1986-12-31"),
    ("AFL", "Aflac Incorporated", "Financials", "Life & Health Insurance", "1999-05-28"),
    ("FTNT", "Fortinet Inc.", "Information Technology", "Systems Software", "2018-10-11"),
    ("NOC", "Northrop Grumman Corporation", "Industrials", "Aerospace & Defense", "1985-06-30"),
    ("HES", "Hess Corporation", "Energy", "Oil & Gas Exploration & Production", "1984-05-31"),
    ("PAYX", "Paychex Inc.", "Industrials", "Human Resource & Employment Services", "1998-10-01"),
    ("FIS", "Fidelity National Information Services Inc.", "Financials", "Transaction & Payment Processing Services", "2006-11-10"),
    ("OKE", "ONEOK Inc.", "Energy", "Oil & Gas Storage & Transportation", "2010-03-12"),
    ("AMP", "Ameriprise Financial Inc.", "Financials", "Asset Management & Custody Banks", "1976-06-30"),
    ("CTSH", "Cognizant Technology Solutions Corporation", "Information Technology", "IT Consulting & Other Services", "2006-11-02"),
    ("DHI", "D.R. Horton Inc.", "Consumer Discretionary", "Homebuilding", "2005-06-22"),
    ("KHC", "The Kraft Heinz Company", "Consumer Staples", "Packaged Foods & Meats", "2015-07-06"),
    ("TRV", "The Travelers Companies Inc.", "Financials", "Property & Casualty Insurance", "2002-08-21"),
    ("PCAR", "PACCAR Inc", "Industrials", "Construction Machinery & Heavy Trucks", "1980-12-31"),
    ("BKR", "Baker Hughes Company", "Energy", "Oilfield Services", "2017-07-07"),
    ("ALL", "The Allstate Corporation", "Financials", "Property & Casualty Insurance", "1995-07-03"),
    ("ILMN", "Illumina Inc.", "Health Care", "Life Sciences Tools & Services", "2015-11-19"),
    ("HAL", "Halliburton Company", "Energy", "Oilfield Services", "1957-03-04"),
    ("YUM", "Yum! Brands Inc.", "Consumer Discretionary", "Restaurants", "1997-10-06"),
    ("VLO", "Valero Energy Corporation", "Energy", "Refining & Marketing", "2002-12-20"),
    ("CARR", "Carrier Global Corporation", "Industrials", "Building Products", "2020-04-03"),
    ("DD", "DuPont de Nemours Inc.", "Materials", "Specialty Chemicals", "1967-09-30"),
    ("RMD", "ResMed Inc.", "Health Care", "Health Care Equipment", "2017-07-26"),
    ("ED", "Consolidated Edison Inc.", "Utilities", "Electric Utilities", "1957-03-04"),
    ("HPQ", "HP Inc.", "Information Technology", "Technology Hardware, Storage & Peripherals", "1974-12-31"),
    ("VRSK", "Verisk Analytics Inc.", "Industrials", "Research & Consulting Services", "2015-02-18"),
    ("GPN", "Global Payments Inc.", "Financials", "Transaction & Payment Processing Services", "2016-04-25"),
    ("EA", "Electronic Arts Inc.", "Communication Services", "Interactive Home Entertainment", "2002-07-22"),
    ("FAST", "Fastenal Company", "Industrials", "Trading Companies & Distributors", "2008-09-15"),
    ("PEG", "Public Service Enterprise Group Incorporated", "Utilities", "Electric Utilities", "1957-03-04"),
    ("WELL", "Welltower Inc.", "Real Estate", "Health Care REITs", "2009-01-30"),
    ("MTD", "Mettler-Toledo International Inc.", "Health Care", "Life Sciences Tools & Services", "2016-09-06"),
    ("AWK", "American Water Works Company Inc.", "Utilities", "Water Utilities", "2016-04-26"),
    ("KEYS", "Keysight Technologies Inc.", "Information Technology", "Test & Measurement Equipment", "2018-11-06"),
    ("DLTR", "Dollar Tree Inc.", "Consumer Discretionary", "General Merchandise Stores", "2011-12-19"),
    ("SBAC", "SBA Communications Corporation", "Real Estate", "Specialized REITs", "2017-09-01"),
    ("GLW", "Corning Incorporated", "Information Technology", "Electronic Components", "1995-02-27"),
    ("WBD", "Warner Bros. Discovery Inc.", "Communication Services", "Movies & Entertainment", "2022-04-11"),
    ("STT", "State Street Corporation", "Financials", "Asset Management & Custody Banks", "2003-03-14"),
    ("WEC", "WEC Energy Group Inc.", "Utilities", "Electric Utilities", "2008-01-02"),
    ("APTV", "Aptiv PLC", "Consumer Discretionary", "Auto Parts & Equipment", "2012-12-24"),
    ("ES", "Eversource Energy", "Utilities", "Multi-Utilities", "2009-07-24"),
    ("ANET", "Arista Networks Inc.", "Information Technology", "Communications Equipment", "2018-08-28"),
    ("EBAY", "eBay Inc.", "Consumer Discretionary", "Broadline Retail", "2002-07-22"),
    ("DFS", "Discover Financial Services", "Financials", "Consumer Finance", "2007-07-02"),
    ("LEN", "Lennar Corporation Class A", "Consumer Discretionary", "Homebuilding", "2005-06-22"),
    ("WY", "Weyerhaeuser Company", "Real Estate", "Timber REITs", "1979-10-01"),
    ("PPG", "PPG Industries Inc.", "Materials", "Specialty Chemicals", "1957-03-04"),
    ("TSCO", "Tractor Supply Company", "Consumer Discretionary", "Specialty Stores", "2014-01-24"),
    ("VMC", "Vulcan Materials Company", "Materials", "Construction Materials", "1999-06-30"),
    ("FITB", "Fifth Third Bancorp", "Financials", "Regional Banks", "1996-03-11"),
    ("ZBH", "Zimmer Biomet Holdings Inc.", "Health Care", "Health Care Equipment", "2001-08-07"),
    ("GWW", "W.W. Grainger Inc.", "Industrials", "Trading Companies & Distributors", "1981-06-30"),
    ("DAL", "Delta Air Lines Inc.", "Industrials", "Airlines", "2013-09-11"),
    ("EIX", "Edison International", "Utilities", "Electric Utilities", "1957-03-04"),
    ("RJF", "Raymond James Financial Inc.", "Financials", "Investment Banking & Brokerage", "2017-03-20"),
    ("LYB", "LyondellBasell Industries N.V.", "Materials", "Commodity Chemicals", "2012-09-05"),
    ("AVB", "AvalonBay Communities Inc.", "Real Estate", "Residential REITs", "2007-01-10"),
    ("EQR", "Equity Residential", "Real Estate", "Residential REITs", "2001-12-03"),
    ("IR", "Ingersoll Rand Inc.", "Industrials", "Industrial Machinery", "2013-05-23"),
    ("MTB", "M&T Bank Corporation", "Financials", "Regional Banks", "2004-02-23"),
    ("HIG", "The Hartford Financial Services Group Inc.", "Financials", "Property & Casualty Insurance", "2008-06-30"),
    ("ARE", "Alexandria Real Estate Equities Inc.", "Real Estate", "Office REITs", "2017-03-20"),
    ("FE", "FirstEnergy Corp.", "Utilities", "Electric Utilities", "1997-11-28"),
    ("BAX", "Baxter International Inc.", "Health Care", "Health Care Equipment", "1972-09-30"),
    ("LH", "Laboratory Corporation of America Holdings", "Health Care", "Health Care Services", "2004-11-01"),
    ("ALGN", "Align Technology Inc.", "Health Care", "Health Care Supplies", "2017-06-19"),
    ("RF", "Regions Financial Corporation", "Financials", "Regional Banks", "1998-08-28"),
    ("ULTA", "Ulta Beauty Inc.", "Consumer Discretionary", "Specialty Stores", "2016-04-18"),
    ("URI", "United Rentals Inc.", "Industrials", "Trading Companies & Distributors", "2014-09-20"),
    ("LUV", "Southwest Airlines Co.", "Industrials", "Airlines", "1994-06-22"),
    ("K", "Kellanova", "Consumer Staples", "Packaged Foods & Meats", "1989-09-11"),
    ("DOV", "Dover Corporation", "Industrials", "Industrial Machinery", "1985-10-31"),
    ("EXR", "Extra Space Storage Inc.", "Real Estate", "Specialized REITs", "2016-01-19"),
    ("HPE", "Hewlett Packard Enterprise Company", "Information Technology", "Technology Hardware, Storage & Peripherals", "2015-11-02"),
    ("VTR", "Ventas Inc.", "Real Estate", "Health Care REITs", "2009-03-04"),
    ("CHTR", "Charter Communications Inc. Class A", "Communication Services", "Cable & Satellite", "2016-09-08"),
    ("AEE", "Ameren Corporation", "Utilities", "Multi-Utilities", "1991-09-19"),
    ("HBAN", "Huntington Bancshares Incorporated", "Financials", "Regional Banks", "2000-04-03"),
    ("NUE", "Nucor Corporation", "Materials", "Steel", "1985-04-30"),
    ("CTRA", "Coterra Energy Inc.", "Energy", "Oil & Gas Exploration & Production", "2008-06-10"),
    ("DTE", "DTE Energy Company", "Utilities", "Multi-Utilities", "1957-03-04"),
    ("ROL", "Rollins Inc.", "Industrials", "Environmental & Facilities Services", "2018-10-01"),
    ("EXPD", "Expeditors International of Washington Inc.", "Industrials", "Air Freight & Logistics", "2007-11-02"),
    ("COO", "The Cooper Companies Inc.", "Health Care", "Health Care Supplies", "2016-11-01"),
    ("MAA", "Mid-America Apartment Communities Inc.", "Real Estate", "Residential REITs", "2016-12-02"),
    ("FSLR", "First Solar Inc.", "Information Technology", "Semiconductors", "2009-03-04"),
    ("HWM", "Howmet Aerospace Inc.", "Industrials", "Aerospace & Defense", "2016-10-21"),
    ("IFF", "International Flavors & Fragrances Inc.", "Materials", "Specialty Chemicals", "1976-03-31"),
    ("CLX", "The Clorox Company", "Consumer Staples", "Household Products", "1969-03-31"),
    ("TDY", "Teledyne Technologies Incorporated", "Industrials", "Aerospace & Defense", "2020-11-05"),
    ("WAT", "Waters Corporation", "Health Care", "Life Sciences Tools & Services", "2002-01-02"),
    ("SWKS", "Skyworks Solutions Inc.", "Information Technology", "Semiconductors", "2015-03-12"),
    ("CCL", "Carnival Corporation", "Consumer Discretionary", "Hotels, Resorts & Cruise Lines", "1998-12-22"),
    ("TYL", "Tyler Technologies Inc.", "Information Technology", "Application Software", "2020-06-22"),
    ("NVR", "NVR Inc.", "Consumer Discretionary", "Homebuilding", "2019-09-26"),
    ("PTC", "PTC Inc.", "Information Technology", "Application Software", "2021-04-20"),
    ("ATO", "Atmos Energy Corporation", "Utilities", "Gas Utilities", "2019-02-15"),
    ("CNP", "CenterPoint Energy Inc.", "Utilities", "Multi-Utilities", "1985-04-30"),
    ("FANG", "Diamondback Energy Inc.", "Energy", "Oil & Gas Exploration & Production", "2018-12-03"),
    ("OMC", "Omnicom Group Inc.", "Communication Services", "Advertising", "1997-12-31"),
    ("AES", "The AES Corporation", "Utilities", "Independent Power Producers & Energy Traders", "1998-10-02"),
    ("IEX", "IDEX Corporation", "Industrials", "Industrial Machinery", "2019-08-09"),
    ("TRMB", "Trimble Inc.", "Information Technology", "Electronic Equipment & Instruments", "2021-01-21"),
    ("WAB", "Westinghouse Air Brake Technologies Corporation", "Industrials", "Construction Machinery & Heavy Trucks", "2019-02-27"),
    ("CBOE", "Cboe Global Markets Inc.", "Financials", "Financial Exchanges & Data", "2017-03-01"),
    ("TER", "Teradyne Inc.", "Information Technology", "Semiconductor Equipment", "2020-09-21"),
    ("STE", "STERIS plc", "Health Care", "Health Care Equipment", "2019-12-23"),
    ("CMS", "CMS Energy Corporation", "Utilities", "Multi-Utilities", "1999-05-03"),
    ("BBY", "Best Buy Co. Inc.", "Consumer Discretionary", "Computer & Electronics Retail", "1999-06-10"),
    ("AMCR", "Amcor plc", "Materials", "Paper Packaging", "2019-06-07"),
    ("MGM", "MGM Resorts International", "Consumer Discretionary", "Casinos & Gaming", "2017-07-26"),
    ("SJM", "The J.M. Smucker Company", "Consumer Staples", "Packaged Foods & Meats", "2008-11-06"),
    ("ETSY", "Etsy Inc.", "Consumer Discretionary", "Broadline Retail", "2020-09-21"),
    ("PODD", "Insulet Corporation", "Health Care", "Health Care Equipment", "2021-03-22"),
    ("NDSN", "Nordson Corporation", "Industrials", "Industrial Machinery", "2022-02-15"),
    ("BALL", "Ball Corporation", "Materials", "Metal, Glass & Plastic Containers", "1984-10-31"),
    ("GRMN", "Garmin Ltd.", "Consumer Discretionary", "Consumer Electronics", "2012-12-12"),
    ("CPB", "Campbell Soup Company", "Consumer Staples", "Packaged Foods & Meats", "1962-09-30"),
    ("EPAM", "EPAM Systems Inc.", "Information Technology", "IT Consulting & Other Services", "2021-12-14"),
    ("RVTY", "Revvity Inc.", "Health Care", "Life Sciences Tools & Services", "1964-03-31"),
    ("MOS", "The Mosaic Company", "Materials", "Fertilizers & Agricultural Chemicals", "2011-09-26"),
    ("BEN", "Franklin Resources Inc.", "Financials", "Asset Management & Custody Banks", "1998-04-30"),
    ("CINF", "Cincinnati Financial Corporation", "Financials", "Property & Casualty Insurance", "1997-12-18"),
    ("KMX", "CarMax Inc.", "Consumer Discretionary", "Automotive Retail", "2010-06-28"),
    ("CPT", "Camden Property Trust", "Real Estate", "Residential REITs", "2022-04-04"),
    ("J", "Jacobs Solutions Inc.", "Industrials", "Construction & Engineering", "1976-06-30"),
    ("FDS", "FactSet Research Systems Inc.", "Financials", "Financial Exchanges & Data", "2021-12-20"),
    ("INCY", "Incyte Corporation", "Health Care", "Biotechnology", "2017-02-28"),
    ("SEDG", "SolarEdge Technologies Inc.", "Information Technology", "Semiconductors", "2021-01-21"),
    ("GEN", "Gen Digital Inc.", "Information Technology", "Systems Software", "2003-03-25"),
    ("UDR", "UDR Inc.", "Real Estate", "Residential REITs", "2016-03-10"),
    ("SWK", "Stanley Black & Decker Inc.", "Industrials", "Industrial Machinery", "1982-09-30"),
    ("HRL", "Hormel Foods Corporation", "Consumer Staples", "Packaged Foods & Meats", "2009-03-04"),
    ("PFG", "Principal Financial Group Inc.", "Financials", "Life & Health Insurance", "2002-07-22"),
    ("BRO", "Brown & Brown Inc.", "Financials", "Insurance Brokers", "2021-06-04"),
    ("JKHY", "Jack Henry & Associates Inc.", "Financials", "Transaction & Payment Processing Services", "2018-03-19"),
    ("EG", "Everest Group Ltd.", "Financials", "Reinsurance", "2017-12-18"),
    ("PKG", "Packaging Corporation of America", "Materials", "Paper Packaging", "2017-07-26"),
    ("NTAP", "NetApp Inc.", "Information Technology", "Technology Hardware, Storage & Peripherals", "1999-06-25"),
    ("DPZ", "Domino's Pizza Inc.", "Consumer Discretionary", "Restaurants", "2020-05-12"),
    ("CDW", "CDW Corporation", "Information Technology", "Technology Distributors", "2019-09-23"),
    ("VFC", "VF Corporation", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods", "1979-06-30"),
    ("AKAM", "Akamai Technologies Inc.", "Information Technology", "Internet Services & Infrastructure", "2007-07-12"),
    ("CZR", "Caesars Entertainment Inc.", "Consumer Discretionary", "Casinos & Gaming", "2021-03-22"),
    ("TFX", "Teleflex Incorporated", "Health Care", "Health Care Equipment", "2019-01-03"),
    ("LDOS", "Leidos Holdings Inc.", "Industrials", "Aerospace & Defense", "2019-11-01"),
    ("EQT", "EQT Corporation", "Energy", "Oil & Gas Exploration & Production", "2022-10-03"),
    ("IP", "International Paper Company", "Materials", "Paper Products", "1957-03-04"),
    ("L", "Loews Corporation", "Financials", "Multi-Sector Holdings", "1995-06-01"),
    ("BG", "Bunge Global SA", "Consumer Staples", "Agricultural Products & Services", "2023-12-18"),
    ("MAS", "Masco Corporation", "Industrials", "Building Products", "1981-06-30"),
    ("TECH", "Bio-Techne Corporation", "Health Care", "Life Sciences Tools & Services", "2021-10-13"),
    ("WYNN", "Wynn Resorts Limited", "Consumer Discretionary", "Casinos & Gaming", "2008-11-14"),
    ("TAP", "Molson Coors Beverage Company Class B", "Consumer Staples", "Brewers", "1976-06-30"),
    ("CRL", "Charles River Laboratories International Inc.", "Health Care", "Life Sciences Tools & Services", "2021-05-14"),
    ("UHS", "Universal Health Services Inc. Class B", "Health Care", "Health Care Facilities", "2014-09-20"),
    ("QRVO", "Qorvo Inc.", "Information Technology", "Semiconductors", "2015-06-11"),
    ("PNR", "Pentair plc", "Industrials", "Industrial Machinery", "2012-10-01"),
    ("HST", "Host Hotels & Resorts Inc.", "Real Estate", "Hotel & Resort REITs", "2007-03-20"),
    ("BXP", "Boston Properties Inc.", "Real Estate", "Office REITs", "2006-04-03"),
    ("ALLE", "Allegion plc", "Industrials", "Building Products", "2013-12-02"),
    ("RL", "Ralph Lauren Corporation", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods", "2007-02-02"),
    ("HSIC", "Henry Schein Inc.", "Health Care", "Health Care Distributors", "2015-03-17"),
    ("AOS", "A. O. Smith Corporation", "Industrials", "Building Products", "2017-07-26"),
    ("NI", "NiSource Inc.", "Utilities", "Multi-Utilities", "1988-01-29"),
    ("REG", "Regency Centers Corporation", "Real Estate", "Retail REITs", "2017-03-02"),
    ("JNPR", "Juniper Networks Inc.", "Information Technology", "Communications Equipment", "2006-06-22"),
    ("FOXA", "Fox Corporation Class A", "Communication Services", "Movies & Entertainment", "2019-03-04"),
    ("FOX", "Fox Corporation Class B", "Communication Services", "Movies & Entertainment", "2019-03-04"),
    ("DAY", "Dayforce Inc.", "Information Technology", "Application Software", "2024-06-24"),
    ("BIO", "Bio-Rad Laboratories Inc. Class A", "Health Care", "Life Sciences Tools & Services", "2020-06-22"),
    ("GL", "Globe Life Inc.", "Financials", "Life & Health Insurance", "1989-04-05"),
    ("MAT", "Mattel Inc.", "Consumer Discretionary", "Leisure Products", "2023-12-18"),
    ("PARA", "Paramount Global Class B", "Communication Services", "Movies & Entertainment", "1994-09-30"),
    ("NWSA", "News Corporation Class A", "Communication Services", "Publishing", "2013-06-06"),
    ("NWS", "News Corporation Class B", "Communication Services", "Publishing", "2013-06-06"),
    ("TPR", "Tapestry Inc.", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods", "2004-09-01"),
    ("IPG", "The Interpublic Group of Companies Inc.", "Communication Services", "Advertising", "1992-10-01"),
    ("BF.B", "Brown-Forman Corporation Class B", "Consumer Staples", "Distillers & Vintners", "1982-10-31"),
    ("MKTX", "MarketAxess Holdings Inc.", "Financials", "Financial Exchanges & Data", "2019-07-01"),
    ("AIZ", "Assurant Inc.", "Financials", "Multi-line Insurance", "2007-04-10"),
    ("HII", "Huntington Ingalls Industries Inc.", "Industrials", "Aerospace & Defense", "2018-01-03"),
    ("LW", "Lamb Weston Holdings Inc.", "Consumer Staples", "Packaged Foods & Meats", "2018-12-03"),
    ("MHK", "Mohawk Industries Inc.", "Consumer Discretionary", "Home Furnishings", "2013-12-23"),
    ("HAS", "Hasbro Inc.", "Consumer Discretionary", "Leisure Products", "1984-09-30"),
    ("CMA", "Comerica Incorporated", "Financials", "Regional Banks", "1995-03-31"),
    ("FRT", "Federal Realty Investment Trust", "Real Estate", "Retail REITs", "2016-03-02"),
    ("XRAY", "Dentsply Sirona Inc.", "Health Care", "Health Care Supplies", "2016-12-23"),
    ("PNW", "Pinnacle West Capital Corporation", "Utilities", "Electric Utilities", "1999-10-01"),
    ("RHI", "Robert Half Inc.", "Industrials", "Human Resource & Employment Services", "2000-12-05"),
    ("ZION", "Zions Bancorporation N.A.", "Financials", "Regional Banks", "2001-06-22"),
    ("NCLH", "Norwegian Cruise Line Holdings Ltd.", "Consumer Discretionary", "Hotels, Resorts & Cruise Lines", "2017-10-13"),
    ("IVZ", "Invesco Ltd.", "Financials", "Asset Management & Custody Banks", "2008-08-21"),
    ("AAL", "American Airlines Group Inc.", "Industrials", "Airlines", "2015-03-23"),
    ("FMC", "FMC Corporation", "Materials", "Fertilizers & Agricultural Chemicals", "2009-08-19"),
    ("WRK", "WestRock Company", "Materials", "Paper Packaging", "2015-07-01"),
    ("CPRI", "Capri Holdings Limited", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods", "2012-02-01"),
    ("RL", "Ralph Lauren Corporation", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods", "2007-02-02"),
    ("DISH", "DISH Network Corporation Class A", "Communication Services", "Cable & Satellite", "2023-06-20"),
    ("GNRC", "Generac Holdings Inc.", "Industrials", "Electrical Components & Equipment", "2021-03-22"),
    ("ALK", "Alaska Air Group Inc.", "Industrials", "Airlines", "2016-05-13"),
    ("DVA", "DaVita Inc.", "Health Care", "Health Care Services", "2008-07-09"),
    ("CE", "Celanese Corporation", "Materials", "Specialty Chemicals", "2018-12-24"),
    ("SNA", "Snap-on Incorporated", "Industrials", "Industrial Machinery", "1982-09-30"),
    ("BWA", "BorgWarner Inc.", "Consumer Discretionary", "Auto Parts & Equipment", "2011-12-19"),
    ("ROL", "Rollins Inc.", "Industrials", "Environmental & Facilities Services", "2018-10-01"),
    ("WHR", "Whirlpool Corporation", "Consumer Discretionary", "Household Appliances", "1967-06-30"),
    ("SEE", "Sealed Air Corporation", "Materials", "Paper Packaging", "1957-03-04"),
    ("ALLE", "Allegion plc", "Industrials", "Building Products", "2013-12-02"),
    ("CHRW", "C.H. Robinson Worldwide Inc.", "Industrials", "Air Freight & Logistics", "2007-03-02"),
    ("FFIV", "F5 Inc.", "Information Technology", "Communications Equipment", "2010-12-20"),
    ("LNC", "Lincoln National Corporation", "Financials", "Life & Health Insurance", "1997-04-30"),
    ("ROL", "Rollins Inc.", "Industrials", "Environmental & Facilities Services", "2018-10-01"),
    ("AOS", "A. O. Smith Corporation", "Industrials", "Building Products", "2017-07-26"),
    ("PNW", "Pinnacle West Capital Corporation", "Utilities", "Electric Utilities", "1999-10-01"),
    ("NWSA", "News Corporation Class A", "Communication Services", "Publishing", "2013-06-06"),
    ("NWS", "News Corporation Class B", "Communication Services", "Publishing", "2013-06-06"),
    ("RL", "Ralph Lauren Corporation", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods", "2007-02-02"),
    ("DISH", "DISH Network Corporation Class A", "Communication Services", "Cable & Satellite", "2023-06-20"),
    ("GNRC", "Generac Holdings Inc.", "Industrials", "Electrical Components & Equipment", "2021-03-22"),
    ("ALK", "Alaska Air Group Inc.", "Industrials", "Airlines", "2016-05-13"),
    ("DVA", "DaVita Inc.", "Health Care", "Health Care Services", "2008-07-09"),
    ("CE", "Celanese Corporation", "Materials", "Specialty Chemicals", "2018-12-24"),
    ("SNA", "Snap-on Incorporated", "Industrials", "Industrial Machinery", "1982-09-30"),
    ("BWA", "BorgWarner Inc.", "Consumer Discretionary", "Auto Parts & Equipment", "2011-12-19"),
    ("WHR", "Whirlpool Corporation", "Consumer Discretionary", "Household Appliances", "1967-06-30"),
    ("SEE", "Sealed Air Corporation", "Materials", "Paper Packaging", "1957-03-04"),
    ("CHRW", "C.H. Robinson Worldwide Inc.", "Industrials", "Air Freight & Logistics", "2007-03-02"),
    ("FFIV", "F5 Inc.", "Information Technology", "Communications Equipment", "2010-12-20"),
    ("LNC", "Lincoln National Corporation", "Financials", "Life & Health Insurance", "1997-04-30"),
    ("ZION", "Zions Bancorporation N.A.", "Financials", "Regional Banks", "2001-06-22"),
    ("NCLH", "Norwegian Cruise Line Holdings Ltd.", "Consumer Discretionary", "Hotels, Resorts & Cruise Lines", "2017-10-13"),
    ("IVZ", "Invesco Ltd.", "Financials", "Asset Management & Custody Banks", "2008-08-21"),
    ("AAL", "American Airlines Group Inc.", "Industrials", "Airlines", "2015-03-23"),
    ("FMC", "FMC Corporation", "Materials", "Fertilizers & Agricultural Chemicals", "2009-08-19"),
    ("WRK", "WestRock Company", "Materials", "Paper Packaging", "2015-07-01"),
    ("CPRI", "Capri Holdings Limited", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods", "2012-02-01"),
]

def create_tables(conn):
    """Create all database tables with proper schema."""
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # 1. tickers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickers (
            ticker TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sector TEXT,
            industry TEXT,
            date_added DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 2. stock_prices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adj_close REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date),
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        )
    """)
    
    # 3. technical_indicators table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            rsi_14 REAL,
            macd_line REAL,
            macd_signal REAL,
            macd_histogram REAL,
            bb_upper REAL,
            bb_middle REAL,
            bb_lower REAL,
            sma_20 REAL,
            sma_50 REAL,
            sma_200 REAL,
            ema_12 REAL,
            ema_26 REAL,
            atr_14 REAL,
            volume_sma_20 REAL,
            price_roc_10 REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date),
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        )
    """)
    
    # 4. news_headlines table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_headlines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            headline TEXT NOT NULL,
            source TEXT,
            url TEXT,
            published_at TIMESTAMP,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 5. sentiment_scores table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_id INTEGER NOT NULL,
            positive_score REAL,
            negative_score REAL,
            neutral_score REAL,
            sentiment_label TEXT,
            confidence REAL,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (news_id) REFERENCES news_headlines(id)
        )
    """)
    
    # 6. daily_sentiment table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            avg_positive REAL,
            avg_negative REAL,
            avg_neutral REAL,
            overall_sentiment TEXT,
            confidence REAL,
            news_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date),
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        )
    """)
    
    # 7. predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            model_name TEXT NOT NULL,
            predicted_direction INTEGER,
            predicted_price REAL,
            confidence REAL,
            features_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        )
    """)
    
    # 8. model_performance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            ticker TEXT,
            train_date_start DATE,
            train_date_end DATE,
            test_date_start DATE,
            test_date_end DATE,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            total_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES tickers(ticker)
        )
    """)
    
    # 9. system_logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT NOT NULL,
            component TEXT NOT NULL,
            message TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 10. user_preferences table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_name TEXT UNIQUE,
            setting_value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 11. schema_migrations table (for migration tracking)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER UNIQUE,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
    """)
    
    conn.commit()
    print("[OK] All 11 tables created successfully")

def create_indexes(conn):
    """Create indexes for query performance."""
    cursor = conn.cursor()
    
    indexes = [
        # Stock prices indexes
        ("idx_stock_prices_ticker", "CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker ON stock_prices(ticker)"),
        ("idx_stock_prices_date", "CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices(date)"),
        ("idx_stock_prices_ticker_date", "CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker_date ON stock_prices(ticker, date)"),
        
        # Technical indicators indexes
        ("idx_technical_indicators_ticker", "CREATE INDEX IF NOT EXISTS idx_technical_indicators_ticker ON technical_indicators(ticker)"),
        ("idx_technical_indicators_date", "CREATE INDEX IF NOT EXISTS idx_technical_indicators_date ON technical_indicators(date)"),
        ("idx_technical_indicators_ticker_date", "CREATE INDEX IF NOT EXISTS idx_technical_indicators_ticker_date ON technical_indicators(ticker, date)"),
        
        # News headlines indexes
        ("idx_news_headlines_ticker", "CREATE INDEX IF NOT EXISTS idx_news_headlines_ticker ON news_headlines(ticker)"),
        ("idx_news_headlines_published", "CREATE INDEX IF NOT EXISTS idx_news_headlines_published ON news_headlines(published_at)"),
        
        # Sentiment scores indexes
        ("idx_sentiment_scores_news_id", "CREATE INDEX IF NOT EXISTS idx_sentiment_scores_news_id ON sentiment_scores(news_id)"),
        
        # Daily sentiment indexes
        ("idx_daily_sentiment_ticker", "CREATE INDEX IF NOT EXISTS idx_daily_sentiment_ticker ON daily_sentiment(ticker)"),
        ("idx_daily_sentiment_date", "CREATE INDEX IF NOT EXISTS idx_daily_sentiment_date ON daily_sentiment(date)"),
        ("idx_daily_sentiment_ticker_date", "CREATE INDEX IF NOT EXISTS idx_daily_sentiment_ticker_date ON daily_sentiment(ticker, date)"),
        
        # Predictions indexes
        ("idx_predictions_ticker", "CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker)"),
        ("idx_predictions_date", "CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(date)"),
        ("idx_predictions_model", "CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name)"),
        
        # Model performance indexes
        ("idx_model_performance_model", "CREATE INDEX IF NOT EXISTS idx_model_performance_model ON model_performance(model_name)"),
        ("idx_model_performance_ticker", "CREATE INDEX IF NOT EXISTS idx_model_performance_ticker ON model_performance(ticker)"),
        
        # System logs indexes
        ("idx_system_logs_level", "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level)"),
        ("idx_system_logs_component", "CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component)"),
        ("idx_system_logs_created", "CREATE INDEX IF NOT EXISTS idx_system_logs_created ON system_logs(created_at)"),
    ]
    
    for name, sql in indexes:
        cursor.execute(sql)
    
    conn.commit()
    print(f"[OK] {len(indexes)} indexes created successfully")

def populate_tickers(conn):
    """Populate tickers table with S&P 500 constituents."""
    cursor = conn.cursor()
    
    cursor.executemany(
        "INSERT OR IGNORE INTO tickers (ticker, name, sector, industry, date_added) VALUES (?, ?, ?, ?, ?)",
        SP500_TICKERS
    )
    
    conn.commit()
    count = cursor.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]
    print(f"[OK] Populated {count} tickers")

def enable_wal_mode(conn):
    """Enable WAL mode for better concurrency."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    result = cursor.fetchone()
    print(f"[OK] WAL mode enabled: {result[0]}")

def record_migration(conn, version, description):
    """Record a schema migration."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO schema_migrations (version, description) VALUES (?, ?)",
        (version, description)
    )
    conn.commit()

def main():
    """Main initialization function."""
    print("=" * 60)
    print("Financial Advisor Bot - Database Initialization")
    print("=" * 60)
    print(f"Database path: {DATABASE_PATH.absolute()}")
    print()
    
    # Ensure data directory exists
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(DATABASE_PATH)
    
    try:
        # Create tables
        create_tables(conn)
        
        # Create indexes
        create_indexes(conn)
        
        # Enable WAL mode
        enable_wal_mode(conn)
        
        # Populate tickers
        populate_tickers(conn)
        
        # Record initial migration
        record_migration(conn, 1, "Initial schema creation with 10 tables")
        
        print()
        print("=" * 60)
        print("[OK] Database initialization complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
