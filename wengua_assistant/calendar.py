"""Source-backed 23:00 day change, isolated from the frozen App calendar.

Both books use a 23:00--01:00 Zi hour and Five-Rat hour stems. Explicit
23:30/23:45 examples in Kaiwu count that Zi hour as the following Ganzhi day.
Use the ACTUAL instant for solar terms/year/month; advance the DAY only.
Every non-23:00 input still takes the original byte-identical casting path.
"""
from datetime import timedelta
from pathlib import Path
import math

SCHEMA = 'wengua_assistant_late_zi_chart_v1'
RULE = 'two_book_zi_initial_23_v1'
RULE_LABEL = '子初23点换日'
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'data/assistant/late_zi_sources.json'


def _clock(iso, zone):
    from src.wengua import astro_calendar as a
    from src.wengua.book_rag import digest, dumps
    local = a._at(iso, zone)
    if local.hour != 23: raise ValueError('23点适配只接收当地23:00至23:59')
    # Keep the exact frozen ephemeris, precision window and boundary guard.
    a._need_ephemeris()
    longitude = a._sun_longitude(a._jd(local))
    precise = a._strict_jieqi_index(local.timestamp())
    if precise is None:
        if min(longitude % 15, 15 - longitude % 15) < .05:
            raise ValueError('Astronomical solar-term boundary within ~1.2-hour uncertainty; outside verified UTC crossing window')
        solar_index = math.floor(((longitude - 270) % 360) / 15)
    else: solar_index = precise[0]
    cycle = a.cycle60()
    day_date = local.date() + timedelta(days=1)
    day = cycle[(cycle.index(a.DAY_ANCHOR_GANZHI) + (day_date - a.DAY_ANCHOR).days) % 60]
    solar_year = local.year - (local.month <= 2 and solar_index in (0, 1, 2))
    year = cycle[(solar_year - 1984) % 60]
    month_index = ((solar_index - 3) % 24) // 2
    first = {'甲': '丙', '己': '丙', '乙': '戊', '庚': '戊', '丙': '庚',
             '辛': '庚', '丁': '壬', '壬': '壬', '戊': '甲', '癸': '甲'}[year[0]]
    month = a.STEMS[(a.STEMS.index(first) + month_index) % 10] + a.BRANCHES[(2 + month_index) % 12]
    hour = a.hour_ganzhi_from_day(day[0], a.hour_branch_from_wall_clock(local.hour))
    result = {'method': a.STRICT_JIEQI_VERSION if precise else a.EPHEMERIS_VERSION,
              'ephemeris_flag': 'offline_swiss_moshier_crossing' if precise else 'offline_Meeus_NOAA_apparent_longitude',
              'solar_term_border_ambiguity_rejected_below_degrees': .05,
              'solar_term_boundary_policy': 'strict_instant_new_inclusive' if precise else 'approximate_guarded_outside_window',
              'last_solar_crossing_utc_ms': precise[1] if precise else None,
              'reference_civil_local': local.isoformat(), 'iana_zone': zone,
              'day_boundary': RULE, 'day_boundary_label': RULE_LABEL,
              'day_cycle_civil_date': day_date.isoformat(),
              'sun_ecliptic_longitude_deg': round(longitude, 9), 'active_solar_term': a.SOLAR_TERMS[solar_index],
              'four_pillars': {'year': year, 'month': month, 'day': day, 'hour': hour},
              'day_cycle_book_anchor': {'civil_date': '1995-06-11', 'day_ganzhi': '癸酉',
                                       'second_original_book_fixture': '1995-08-13 丙子',
                                       'third_original_book_fixture': '2001-08-20 乙卯'}}
    result['sha256'] = digest(dumps(result))
    result['note'] = 'Actual instant solar terms; day/hour use the two-book 23:00 convention'
    return result


def cast_time(iso, zone):
    from src.wengua import astro_calendar as a, book_rag_query as old
    from src.wengua import book_rag as r
    from src.wengua.casting_shenqi import bureau_for_term, layout_from_explicit_bureau
    if a._at(iso, zone).hour != 23: return old.cast_time(iso, zone)
    clock = _clock(iso, zone); pillars = clock['four_pillars']
    bureau = bureau_for_term(pillars['day'], clock['active_solar_term'])
    chart = layout_from_explicit_bureau(dun=bureau['dun'], ju=bureau['ju'], hour_ganzhi=pillars['hour'])
    files = {p: r.digest((r.ROOT / p).read_bytes()) for p in old.CAST_FILES}
    files['wengua_assistant/calendar.py'] = r.digest(Path(__file__).read_bytes())
    files['data/assistant/late_zi_sources.json'] = r.digest(SOURCES.read_bytes())
    result = {'schema': SCHEMA, 'method_id': r.METHOD, 'time': iso, 'timezone': zone,
              'calendar': clock, 'bureau': bureau, 'chart': chart, 'implementation_sha256': files,
              'casting_backend': 'existing_app_chai_bu_functions_with_two_book_23h_calendar_adapter',
              'created_by_existing_program': True, 'model_did_not_cast': True,
              'day_boundary_source_rule': r.read(SOURCES)}
    return result | {'receipt_sha256': r.digest(r.dumps(result))}


def checked_chart(packet):
    from src.wengua.book_rag_query import checked_chart as original
    if not isinstance(packet, dict) or packet.get('schema') != SCHEMA: return original(packet)
    if packet != cast_time(packet['time'], packet['timezone']):
        raise ValueError('23点程序盘与真实时间、换日约定或校验记录不一致')
    return packet


def compact_chart(packet):
    from src.wengua.book_rag_query import compact_chart as original
    text = original(packet)
    if packet.get('schema') == SCHEMA:
        text += '\n换日约定：' + RULE_LABEL + '；干支日归属' + packet['calendar']['day_cycle_civil_date'] + '。'
        text += '显示的起局时间保持真实时刻；节气、年柱、月柱按真实交节瞬间计算。'
    return text
