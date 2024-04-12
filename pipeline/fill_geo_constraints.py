import json
import re
from collections import defaultdict
from copy import deepcopy


def _clean_region_name(og_name):
    suffixes = ['(pv)', '(ld)', '(cr)', '(gemeente)', '(drostambt)', '(oud)']
    name = og_name.lower()  # Lowercase the name
    # Remove suffixes
    for suffix in suffixes:
        if suffix in name:
            name = name.replace(suffix, '')
    # Remove additions to the name e.g. (zh.), (l.) etc. Usually provinces for disambiguation
    name = re.sub('\([\w.-]+\)', '', name).strip()

    # Extend vocab with _ and remove -
    region_names = [name]
    if " " in name:
        region_names.append(name.replace(" ", "_"))
    elif "-" in name:
        region_names.append(name.replace("-", "_"))
        region_names.append(name.replace("-", " "))
    return region_names


with open('data/geolocations.json', 'r') as geo_f:
    geos = json.load(geo_f)

    geolocations = defaultdict(list)
    geolocations['nederland'].append('NL01')  # Nederland is not in the json for some reason
    geolocations['friesland'].append('PV22')  # The regular spelling of Friesland is not in the json
    for i in range(0, len(geos["Regio's"])):
        region_names = _clean_region_name(geos["Regio's"][i])
        for region_name in region_names:
            region_code = geos['Gebieds- of gemeentecode (code)'][i].strip()
            geolocations[region_name].append(region_code)


region_pattern = re.compile(f'(?<!\w)({"|".join(geolocations.keys())})(?!\w)')


def match_region(query):
    def _meager_disambiguation_attempt(prompt, candidates):
        """
            Disambiguate based on a couple of keyword hints
        """
        hints = {'corop': 'CR', 'omgeving': 'CR', 'provincie': 'PV', 'gemeente': 'GM', 'stad': 'GM', 'plaats': 'GM'}
        for key in hints.keys():
            if key in prompt.lower():
                for c in candidates:
                    if hints[key] in c:
                        return [c]
        return candidates

    regions = region_pattern.findall(query.lower().strip().replace("-", "_"))
    if regions:
        region_codes = deepcopy(geolocations[regions[0]])  # make sure to copy the values, not references
        if len(region_codes) > 1:
            region_codes = _meager_disambiguation_attempt(query, region_codes)
        return region_codes
    return []
