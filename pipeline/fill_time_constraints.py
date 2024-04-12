import datetime
import re

months = ['januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli',
          'augustus', 'september', 'oktober', 'november', 'december']

months_re = re.compile(fr'(?<!\w)({"|".join(months)})(?!\w)')

year_parts = {'kwartaal': 'KW', 'kwart': 'KW', 'seizoen': 'KW', 'eind': ['KW04', 'HJ02'], 'begin': ['KW01', 'HJ01'],
              'helft': 'HJ', 'q1': 'KW01', 'q2': 'KW02', 'q3': 'KW03', 'q4': 'KW04', 'lente': 'KW01',
              'zomer': 'KW02', 'herfst': 'KW03', 'winterseizoen': 'KW04', 'lenteseizoen': 'KW01',
              'zomerseizoen': 'KW02', 'herfstseizoen': 'KW03', 'winter': 'KW04'}

year_parts_re = re.compile(fr'(?<!\w)({"|".join(year_parts)})(?!\w)')

cardinals = {
    'eerste': '01',
    '1e': '01',
    '1': '01',
    'tweede': '02',
    '2e': '02',
    '2': '02',
    'derde': '03',
    '3e': '03',
    '3': '03',
    'vierde': '04',
    '4e': '04',
    '4': '04',
    'laatste': '04'
}

cardinals_re = re.compile(fr'(?<!\w)({"|".join(cardinals.keys())})(?!\w)')

years_re = re.compile(r'(?<!\d)\d{4}(?!\d)')


def month_to_code(month):
    index = months.index(month) + 1
    if index < 10:
        return f"0{index}"
    return index


def extract_tc(query: str, available_time_constraints: list = None) -> list:
    query = query.lower()

    def _extract_years() -> list:
        years_in_query = years_re.findall(query)
        max_year = datetime.datetime.now().year + 1000
        years_in_query = [year for year in years_in_query if int(year) <= max_year]
        return years_in_query

    years_in_query = list(set(_extract_years()))
    if len(years_in_query) > 1:
        return []

    months_in_query = list(set(months_re.findall(query)))
    if len(months_in_query) > 1:
        return []

    if months_in_query:
        if len(months_in_query) == 1 and len(years_in_query) == 1:
            return [f"{years_in_query[0]}MM{month_to_code(months_in_query[0])}"]

    year_parts_in_query = list(set(year_parts_re.findall(query)))
    if len(year_parts_in_query) > 1:
        return []

    if year_parts_in_query:
        yp = year_parts[year_parts_in_query[0]]
        if isinstance(yp, list):
            if not available_time_constraints or f"{years_in_query[0]}{yp[0]}" in available_time_constraints:
                return [f"{years_in_query[0]}{yp[0]}"]
            if available_time_constraints and f"{years_in_query[0]}{yp[1]}" in available_time_constraints:
                return [f"{years_in_query[0]}{yp[1]}"]
            return []

        if len(yp) == 4:
            return [f"{years_in_query[0]}{yp}"]

        cardinals_in_query = cardinals_re.findall(query)
        cardinal = cardinals[cardinals_in_query[0]]

        return [f"{years_in_query[0]}{year_parts[year_parts_in_query[0]]}{cardinal}"]

    if years_in_query:
        return [f"{years_in_query[0]}JJ00"]

    return []
