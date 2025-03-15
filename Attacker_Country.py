import pandas as pd
import pycountry
from countryinfo import CountryInfo

df = pd.read_parquet("new_train_data.parquet")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)


encoksaldiriyapanulke = df["attacker_country"].value_counts().idxmax()
df["attacker_country"] = df["attacker_country"].fillna(encoksaldiriyapanulke)

attacker_countries = df["attacker_country"].unique().tolist()

country_to_continent = {
    "US": "North America", "CA": "North America", "MX": "North America", "BR": "South America", "AR": "South America",
    "CL": "South America", "PE": "South America", "CO": "South America", "VE": "South America", "BO": "South America",
    "PY": "South America", "UY": "South America", "FR": "Europe", "DE": "Europe", "IT": "Europe",
    "GB": "Europe", "NL": "Europe", "SE": "Europe", "UA": "Europe", "RU": "Europe",
    "CH": "Europe", "FI": "Europe", "ES": "Europe", "BG": "Europe", "PL": "Europe",
    "BE": "Europe", "AT": "Europe", "DK": "Europe", "NO": "Europe", "IE": "Europe",
    "GR": "Europe", "PT": "Europe", "CZ": "Europe", "SK": "Europe", "RO": "Europe",
    "HU": "Europe", "RS": "Europe", "SI": "Europe", "LT": "Europe", "LV": "Europe",
    "BY": "Europe", "HR": "Europe", "BA": "Europe", "AL": "Europe", "XK": "Europe",
    "MD": "Europe", "LU": "Europe", "IS": "Europe", "EE": "Europe", "MK": "Europe",
    "MT": "Europe", "AD": "Europe", "MC": "Europe", "CN": "Asia", "IN": "Asia",
    "SG": "Asia", "JP": "Asia", "KR": "Asia", "VN": "Asia", "HK": "Asia", "TW": "Asia",
    "TR": "Asia", "IR": "Asia", "BD": "Asia", "PH": "Asia", "TH": "Asia", "MY": "Asia",
    "ID": "Asia", "KZ": "Asia", "GE": "Asia", "UZ": "Asia", "AZ": "Asia", "IQ": "Asia",
    "SA": "Asia", "AE": "Asia", "JO": "Asia", "QA": "Asia", "OM": "Asia", "KW": "Asia",
    "LA": "Asia", "LB": "Asia", "PS": "Asia", "AF": "Asia", "PK": "Asia", "AM": "Asia",
    "ZA": "Africa", "EG": "Africa", "NG": "Africa", "DZ": "Africa", "KE": "Africa", "MA": "Africa",
    "GH": "Africa", "ET": "Africa", "TZ": "Africa", "UG": "Africa", "RW": "Africa", "CI": "Africa",
    "SN": "Africa", "BI": "Africa", "ZW": "Africa", "CM": "Africa", "TG": "Africa", "AO": "Africa",
    "NA": "Africa", "MW": "Africa", "ML": "Africa", "AU": "Oceania", "NZ": "Oceania",
    "PG": "Oceania", "FJ": "Oceania", "SB": "Oceania", "GG": "Europe", "JE": "Europe",
    "RE": "Africa", "MQ": "North America", "FO": "Europe", "TN": "Africa", "IL": "Asia",
    "MN": "Asia", "KH": "Asia", "BZ": "North America", "EC": "South America", "GT": "North America",
    "BH": "Asia", "ME": "Europe", "LK": "Asia", "DO": "North America", "NI": "North America",
    "GA": "Africa", "LI": "Europe", "BW": "Africa", "LY": "Africa", "NP": "Asia", "SY": "Asia",
    "CR": "North America", "PF": "Oceania", "CY": "Europe", "JM": "North America", "GU": "Oceania",
    "BS": "North America", "IM": "Europe", "SO": "Africa", "MU": "Africa", "PA": "North America",
    "LC": "North America", "HN": "North America", "AG": "North America", "SR": "South America", "TT": "North America",
    "KG": "Asia", "SC": "Africa", "SV": "North America", "MV": "Asia", "VC": "North America",
    "ZM": "Africa", "MZ": "Africa", "TJ": "Asia", "KM": "Africa", "WS": "Oceania", "BM": "North America",
    "BB": "North America", "BN": "Asia", "MR": "Africa", "GQ": "Africa", "MO": "Asia", "MM": "Asia",
    "PR": "North America", "GN": "Africa", "KN": "North America", "CD": "Africa", "DM": "North America",
    "BJ": "Africa", "YE": "Asia", "BT": "Asia", "VG": "North America", "SS": "Africa",
    "TC": "North America", "BF": "Africa", "KY": "North America", "CV": "Africa", "DJ": "Africa",
    "PM": "North America", "BQ": "North America", "GP": "North America", "HT": "North America", "GW": "Africa",
    "CW": "North America", "GY": "South America", "SD": "Africa", "MG": "Africa", "GM": "Africa",
    "FM": "Oceania", "MP": "Oceania", "LS": "Africa", "CX": "Oceania", "GI": "Europe", "SH": "Africa",
    "CG": "Africa", "SX": "North America", "KP": "Asia", "AW": "North America", "AI": "North America",
    "ST": "Africa", "LR": "Africa", "SZ": "Africa", "VI": "North America", "GF": "South America",
}

df["attacker_continent"] = df["attacker_country"].map(country_to_continent)


# continent sütunundaki boş değerlere karşılık gelen 'attacker_country' değerlerini gösterin
missing_countries_codes = (df[df["attacker_continent"].isna()]["attacker_country"]).unique().tolist()


attacker_continent = pd.get_dummies(df["attacker_continent"], prefix="continent_attack",dtype=int)
attacker_continent.to_parquet("w_continent.parquet")
df = pd.concat([df, attacker_continent], axis=1)
df.drop(["attacker_country","attacker_continent"], axis=1,inplace=True)




