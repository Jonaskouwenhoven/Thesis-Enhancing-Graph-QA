import pandas as pd
from rdflib import URIRef

from odata_graph.ttl_to_py.defined_namespace_generator import generate_defined_namespace
from utils.global_functions import secure_request
from odata_graph.namespaces._QUDT_UNIT import QUDT_UNIT

UNIT = URIRef('http://qudt.org/2.1/schema/qudt/unit')
UNIT_OF_SYSTEM = URIRef('http://qudt.org/2.1/schema/qudt/unitOfSystem')
MULTIPLIER = URIRef('http://qudt.org/2.1/schema/qudt/conversionMultiplier')
RATIO_SCALE = URIRef('http://qudt.org/2.1/schema/qudt/RatioScale')

# Conversion dictionary for OData4 units to QUDT types
# TODO: model units that are not in QUDT
UNIT_DICT = {
    "x 1 000": {"unit": QUDT_UNIT.NUM, "multiplier": 1000},
    "euro": {"unit": QUDT_UNIT.Euro, "multiplier": None},
    "uur": {"unit": QUDT_UNIT.HR, "multiplier": None},
    "%": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "% ": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "% van werkzame beroepsbevolking": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "uren per week": {"unit": QUDT_UNIT.PER_WK, "multiplier": None},
    "jaren": {"unit": QUDT_UNIT.YR, "multiplier": None},
    "% van werknemers": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "% van zelfstandigen": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "x 1000": {"unit": QUDT_UNIT.NUM, "multiplier": 1000},
    "Aantal": {"unit": QUDT_UNIT.NUM, "multiplier": None},
    "aantal": {"unit": QUDT_UNIT.NUM, "multiplier": None},
    "mln euro": {"unit": QUDT_UNIT.Euro, "multiplier": 1000000},
    "% van werkzame personen": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "% van bedrijven": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "Aandeel van bedrijven met afgeronde innovaties": {"unit": None, "multiplier": None},
    "% (van het bbp)": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "per 1 000 inwoners": {"unit": None, "multiplier": None},
    "per 1 000": {"unit": None, "multiplier": None},
    "per 1 000 meisjes": {"unit": None, "multiplier": None},
    "per 1 000 geboren kinderen": {"unit": None, "multiplier": None},
    "jaar": {"unit": QUDT_UNIT.YR, "multiplier": None},
    "per 1 000 ": {"unit": None, "multiplier": None},
    "0/00": {"unit": None, "multiplier": None},
    "per 1 000 levend geborenen": {"unit": None, "multiplier": None},
    "per 1 000 geborenen": {"unit": None, "multiplier": None},
    "Jaar": {"unit": QUDT_UNIT.YR, "multiplier": None},
    "1 000 euro": {"unit": QUDT_UNIT.Euro, "multiplier": 1000},
    "m³": {"unit": QUDT_UNIT.M3, "multiplier": None},
    "m²": {"unit": QUDT_UNIT.M2, "multiplier": None},
    "per 100 000 inwoners": {"unit": None, "multiplier": None},
    "schaalscore": {"unit": None, "multiplier": None},
    "o/oo": {"unit": None, "multiplier": None},
    "per 10 000 personen": {"unit": None, "multiplier": None},
    "dagen": {"unit": QUDT_UNIT.DAY, "multiplier": None},
    "miljard euro": {"unit": QUDT_UNIT.Euro, "multiplier": 1000000000},
    "km": {"unit": QUDT_UNIT.KiloM, "multiplier": None},
    "Ginicoëfficiënt": {"unit": None, "multiplier": None},
    "mln kg": {"unit": QUDT_UNIT.KiloGM, "multiplier": 1000000},
    "x 1 000 arbeidsjaren": {"unit": None, "multiplier": None},
    "2015=100": {"unit": None, "multiplier": None},
    "in % bbp": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "aantal x 1 000": {"unit": QUDT_UNIT.NUM, "multiplier": 1000},
    "mln uren": {"unit": QUDT_UNIT.HR, "multiplier": 1000000},
    "1000 ton": {"unit": QUDT_UNIT.KiloTON_Metric, "multiplier": None},
    "in % van  inwoners 15 jaar of ouder": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "aantal inwoners per km²": {"unit": QUDT_UNIT.NUM_PER_KiloM2, "multiplier": None},
    "personen per 1 huishouden": {"unit": None, "multiplier": None},
    "per 1 000 woningen": {"unit": None, "multiplier": None},
    "aantal woningen per km²": {"unit": QUDT_UNIT.NUM_PER_KiloM2, "multiplier": None},
    "are": {"unit": QUDT_UNIT.ARE, "multiplier": None},
    "kg/ha": {"unit": QUDT_UNIT.KiloGM_PER_HA, "multiplier": None},
    "kilometer": {"unit": QUDT_UNIT.KiloM, "multiplier": None},
    "1 000 kg": {"unit": QUDT_UNIT.KiloM, "multiplier": 1000},
    "km²": {"unit": QUDT_UNIT.M2, "multiplier": 0.000001},
    "per km²": {"unit": QUDT_UNIT.NUM_PER_KiloM2, "multiplier": None},
    "ha": {"unit": QUDT_UNIT.HA, "multiplier": None},
    "in % oppervlakte land": {"unit": None, "multiplier": None},
    "ha / 1 000 inwoners": {"unit": None, "multiplier": None},
    "code": {"unit": None, "multiplier": None},
    "naam": {"unit": None, "multiplier": None},
    "1 000 ton": {"unit": QUDT_UNIT.KiloTON_Metric, "multiplier": None},
    "x mln": {"unit": QUDT_UNIT.NUM, "multiplier": 1000000},
    "mln km": {"unit": QUDT_UNIT.KiloM, "multiplier": 1000000},
    "mln tonkm": {"unit": None, "multiplier": None},
    "mln ": {"unit": QUDT_UNIT.NUM, "multiplier": 1000000},
    "miljoenen US-dollar": {"unit": QUDT_UNIT.USDollar, "multiplier": 1000000},
    "US-dollar": {"unit": QUDT_UNIT.USDollar, "multiplier": None},
    "% van het bbp": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "% van totale export goederen": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "miljoenen personen": {"unit": QUDT_UNIT.NUM, "multiplier": 1000000},
    "absoluut": {"unit": QUDT_UNIT.NUM, "multiplier": None},
    "maanden": {"unit": QUDT_UNIT.MO, "multiplier": None},
    "euro per m²": {"unit": None, "multiplier": None},
    "PJ": {"unit": QUDT_UNIT.PetaJ, "multiplier": None},
    "1998=100": {"unit": None, "multiplier": None},
    "per 10 000 personen in de bevolking": {"unit": None, "multiplier": None},
    "1985=100": {"unit": None, "multiplier": None},
    "ab_so_luut": {"unit": QUDT_UNIT.NUM, "multiplier": None},
    "1989=100": {"unit": None, "multiplier": None},
    "rangnummer": {"unit": None, "multiplier": None},
    "mld kg": {"unit": QUDT_UNIT.KiloGM, "multiplier": 1000000000},
    "1 000 ha": {"unit": QUDT_UNIT.HA, "multiplier": 1000},
    "2010 = 100": {"unit": None, "multiplier": None},
    "% bbp": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "% van bruto alternatief beschikbaar inkomen": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
    "1000 arbeidsjaren": {"unit": None, "multiplier": None},
    "1 000 zware metaal-equivalenten": {"unit": None, "multiplier": None},
    "1 000 nutriënten-equivalenten": {"unit": None, "multiplier": None},
    "mln broeikasgas-equivalenten": {"unit": None, "multiplier": None},
    "mln zuur-equivalenten": {"unit": None, "multiplier": None},
    "1 000 CFK12-equivalenten": {"unit": None, "multiplier": None},
    "1 000 arbeidsjaren": {"unit": None, "multiplier": None},
    "mln m3": {"unit": QUDT_UNIT.M3, "multiplier": 1000000},
    "mld zuur-equivalenten": {"unit": None, "multiplier": None},
    "mld euro": {"unit": QUDT_UNIT.Euro, "multiplier": 1000000000},
    "1000 euro": {"unit": QUDT_UNIT.Euro, "multiplier": 1000},
    "x 1 000 euro": {"unit": QUDT_UNIT.Euro, "multiplier": 1000},
    "index (2005 = 100)": {"unit": None, "multiplier": None},
    "x 1000 euro": {"unit": QUDT_UNIT.Euro, "multiplier": 1000},
    "1990=100": {"unit": None, "multiplier": None},
    "x mld. euro": {"unit": QUDT_UNIT.Euro, "multiplier": 1000000000},
    "euro's": {"unit": QUDT_UNIT.Euro, "multiplier": None},
    "kWh": {"unit": QUDT_UNIT.KiloW_HR, "multiplier": None},
    "per huishouden": {"unit": None, "multiplier": None},
    "euro/inwoner": {"unit": None, "multiplier": None},
    "In % bepaalde leeftijdsklasse": {"unit": QUDT_UNIT.PERCENT, "multiplier": None},
}


if __name__ == "__main__":
    tables = secure_request('https://odata4.cbs.nl/CBS/datasets', json=True, max_retries=3, timeout=3)['value']
    kerncijfer_tables = [t['Identifier'] for t in tables
                         if ('kerncijfers' in t['Title'].lower() and t['Status'] == 'Regulier')]

    csv = pd.read_csv('.\ODataProperties.csv', delimiter=';')
    df = csv[csv['Identifier'].isin(kerncijfer_tables)][['Identifier', 'Type', 'Title', 'DataType', 'Unit']]
    df = df[~df['Unit'].isnull()]

    units = df['Unit'].value_counts()

    generate_defined_namespace(ttl_file='./odata_graph/ttl_to_py/qudt_unit.ttl',
                               py_file='./odata_graph/namespaces/_QUDT_UNIT.py',
                               ns_uri='http://qudt.org/schema/qudt/Unit')
