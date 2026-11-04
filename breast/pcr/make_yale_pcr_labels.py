#!/usr/bin/env python3
"""Build Yale trastuzumab pCR labels: TCIA HER2-tumor-rois V3 clinical XLSX -> CSV.
Source: Farahmand et al. 2022, DOI 10.7937/E65C-AM96 (CC BY 4.0).
The XLSX is strict OOXML (not supported by openpyxl), so the XML is parsed directly.
Usage: python make_yale_pcr_labels.py <metadata.xlsx> <out.csv>
"""
import sys, zipfile, re, csv
import xml.etree.ElementTree as ET

def parse_strict_xlsx(path):
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    ns = root.tag.split('}')[0].strip('{')
    ss = [''.join(t.text or '' for t in si.iter() if t.tag.endswith('}t'))
          for si in root.findall(f'{{{ns}}}si')]
    sh = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    nsh = sh.tag.split('}')[0].strip('{')
    def val(c):
        v = c.find(f'{{{nsh}}}v')
        if v is not None:
            return ss[int(v.text)] if c.get('t') == 's' else v.text
        return ''
    rows = []
    for row in sh.find(f'{{{nsh}}}sheetData').findall(f'{{{nsh}}}row'):
        d = {}
        for c in row.findall(f'{{{nsh}}}c'):
            d[re.match(r'[A-Z]+', c.get('r')).group()] = val(c)
        rows.append(d)
    return rows

def main(xlsx, out):
    rows = parse_strict_xlsx(xlsx)
    hdr = {v: k for k, v in rows[0].items()}
    pcol, rcol = hdr['Patient'], hdr['Responder']
    recs = []
    for r in rows[1:]:
        pid = (r.get(pcol) or '').strip()
        resp = (r.get(rcol) or '').strip().lower()
        if not pid:
            continue
        pcr = 1 if resp == 'responder' else 0
        recs.append((pid, pcr, resp))
    pos = sum(p for _, p, _ in recs)
    assert len(recs) == 85 and pos == 36, f"expected 85 rows / 36 responders, got {len(recs)}/{pos}"
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case_id', 'pcr', 'cohort', 'responder_raw'])
        for pid, pcr, resp in recs:
            w.writerow([pid, pcr, 'Yale_trastuzumab_response_cohort', resp])
    print(f"wrote {out}: {len(recs)} rows (responder {pos}/non {len(recs)-pos})")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
