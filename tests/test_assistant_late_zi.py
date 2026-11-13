"""Calendar boundary regressions, not model or real-outcome accuracy tests."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import hashlib, json, unittest
from wengua_assistant import calendar as new, prompt


class LateZiCalendar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wengua_course.runtime import configure
        try: configure()
        except ValueError: raise unittest.SkipTest('Frozen App calendar absent in this software-only job')
        from src.wengua import astro_calendar, book_rag_query
        cls.old = book_rag_query; cls.astro = astro_calendar

    def test_23_45_matches_original_book_pillars_and_ju_without_changing_real_time(self):
        packet = prompt.cast('2000-09-21T23:45', 'Asia/Shanghai')
        self.assertEqual(packet['schema'], new.SCHEMA)
        self.assertEqual(packet['time'], '2000-09-21T23:45:00+08:00')
        self.assertEqual(packet['calendar']['reference_civil_local'], packet['time'])
        self.assertEqual(packet['calendar']['day_cycle_civil_date'], '2000-09-22')
        self.assertEqual(packet['calendar']['four_pillars'], {'year': '庚辰', 'month': '乙酉', 'day': '癸未', 'hour': '壬子'})
        self.assertEqual((packet['chart']['dun'], packet['chart']['ju']), ('yin', 9))
        self.assertEqual(new.checked_chart(packet), packet)
        text = new.compact_chart(packet)
        self.assertIn('2000-09-21 23:45:00', text)
        self.assertIn('子初23点换日', text)

    def test_23_30_matches_second_original_book_example(self):
        packet = prompt.cast('2000-06-19T23:30', 'Asia/Shanghai')
        self.assertEqual(packet['calendar']['four_pillars'], {'year': '庚辰', 'month': '壬午', 'day': '己酉', 'hour': '甲子'})
        self.assertEqual((packet['chart']['dun'], packet['chart']['ju']), ('yang', 6))
        self.assertEqual((packet['chart']['value_star'], packet['chart']['value_star_palace']), ('天心星', 6))
        self.assertEqual((packet['chart']['value_door'], packet['chart']['value_door_palace']), ('开门', 6))

    def test_22_59_23_00_and_midnight_are_exact_and_legacy_still_refuses_23(self):
        before = prompt.cast('2000-09-21T22:59:59.999999', 'Asia/Shanghai')
        after = prompt.cast('2000-09-21T23:00:00', 'Asia/Shanghai')
        last = prompt.cast('2000-09-21T23:59:59.999999', 'Asia/Shanghai')
        midnight = prompt.cast('2000-09-22T00:00:00', 'Asia/Shanghai')
        self.assertEqual(before['calendar']['four_pillars']['day'], '壬午')
        self.assertEqual(after['calendar']['four_pillars']['day'], '癸未')
        self.assertEqual(after['chart'], last['chart'])
        self.assertEqual(after['chart'], midnight['chart'])
        with self.assertRaisesRegex(ValueError, 'Late-Zi'):
            self.old.cast_time('2000-09-21T23:45:00+08:00', 'Asia/Shanghai')

    def test_all_other_hours_reuse_identical_old_packets_in_three_zones(self):
        for zone in ['Asia/Shanghai', 'America/New_York', 'UTC']:
            for hour in range(23):
                value = f'2026-11-12T{hour:02}:17:31'
                iso = prompt.aware_time(value, zone)
                self.assertEqual(prompt.cast(value, zone), self.old.cast_time(iso, zone))

    def test_ganzhi_cycle_month_end_year_end_and_leap_day(self):
        for date in ['2000-02-28', '2000-02-29', '2026-12-31', '2026-11-12']:
            after = prompt.cast(date + 'T23:30', 'Asia/Shanghai')
            tomorrow = datetime.fromisoformat(date) + timedelta(days=1)
            next_clock = self.astro.solar_chart_clock(tomorrow.isoformat() + '+08:00', 'Asia/Shanghai')
            self.assertEqual(after['calendar']['four_pillars']['day'], next_clock['four_pillars']['day'])
            self.assertEqual(after['calendar']['four_pillars']['hour'], next_clock['four_pillars']['hour'])
        for offset in range(60):
            day = datetime(2026, 10, 1) + timedelta(days=offset)
            local = day.strftime('%Y-%m-%dT23:30:00+08:00')
            clock = new._clock(local, 'Asia/Shanghai')
            tomorrow = self.astro.solar_chart_clock((day + timedelta(days=1)).isoformat() + '+08:00', 'Asia/Shanghai')
            self.assertEqual(clock['four_pillars']['day'], tomorrow['four_pillars']['day'])
            self.assertEqual(clock['four_pillars']['hour'], tomorrow['four_pillars']['hour'])

    def test_lichun_crossing_during_local_23_uses_exact_real_instant(self):
        from zoneinfo import ZoneInfo
        a = self.astro
        crossings = [(ms, (1 + n) % 24) for n, ms in enumerate(a._strict_crossing_times())]
        ms = next(ms for ms, index in crossings if index == 3 and datetime.fromtimestamp(ms / 1000, timezone.utc).year == 2026)
        instant = datetime.fromtimestamp(ms / 1000, timezone.utc)
        delta = (23 - instant.hour) % 24
        if delta > 12: delta -= 24
        zone = 'Etc/GMT' + ('-' if delta > 0 else '+') + str(abs(delta)) if delta else 'UTC'
        local = instant.astimezone(ZoneInfo(zone)); self.assertEqual(local.hour, 23)
        old = new._clock((local - timedelta(milliseconds=1)).isoformat(), zone)
        now = new._clock(local.isoformat(), zone)
        self.assertEqual((old['active_solar_term'], now['active_solar_term']), ('大寒', '立春'))
        self.assertEqual((old['four_pillars']['year'], now['four_pillars']['year']), ('乙巳', '丙午'))
        self.assertEqual((old['four_pillars']['month'][-1], now['four_pillars']['month'][-1]), ('丑', '寅'))
        self.assertEqual(old['four_pillars']['day'], now['four_pillars']['day'])
        self.assertEqual(old['four_pillars']['hour'], now['four_pillars']['hour'])

    def test_precision_and_dst_guards_are_not_bypassed(self):
        with patch.object(self.astro, '_strict_jieqi_index', return_value=None), \
             patch.object(self.astro, '_sun_longitude', return_value=315.001):
            with self.assertRaisesRegex(ValueError, 'solar-term boundary'):
                prompt.cast('2026-11-12T23:30', 'Asia/Shanghai')
        with self.assertRaises(ValueError): prompt.cast('2026-11-01T01:30', 'America/New_York')

    def test_modified_program_chart_or_rule_is_rejected(self):
        packet = prompt.cast('2000-09-21T23:45', 'Asia/Shanghai')
        for mutate in [lambda p: p['calendar']['four_pillars'].update(day='壬午'),
                       lambda p: p.update(time='2000-09-22T00:45:00+08:00'),
                       lambda p: p['day_boundary_source_rule'].update(day_changes_at_local='00:00:00')]:
            changed = deepcopy(packet); mutate(changed)
            with self.assertRaises(ValueError): new.checked_chart(changed)

    def test_source_selectors_match_original_private_books_when_installed(self):
        from wengua_course_v54.corpus import source_library, prior
        if not (prior.DATA / 'rag_sources.json.gz').exists(): self.skipTest('Private original books are not shipped to CI')
        chapters = {(c['book_id'], c['chapter_id']): c for c in source_library()['chapters']}
        for source in json.loads(new.SOURCES.read_text(encoding='utf8'))['sources']:
            chapter = chapters[source['book_id'], source['chapter']]
            self.assertEqual(source['source_sha256'], chapter['source_sha256'])
            text = '\n'.join(chapter['original_text'].splitlines()[source['lines'][0] - 1:source['lines'][1]])
            self.assertEqual(source['quote_sha256'], hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__': unittest.main()
