import os, sys, gzip, random, csv, json, datetime, re
import requests
from bs4 import BeautifulSoup as bs
import urllib3
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']

_URL = "http://dauvi.org/Violino/Bolla/"
_URL = "https://www.bushgrafts.com/jazz/Midi%20site/"

# functional
r = requests.get(_URL)
soup = bs(r.text)
urls = []
names = []
for i, link in enumerate(soup.findAll('a')):
    _FULLURL = _URL + link.get('href',"lxml")
    if _FULLURL.endswith('.mid'):
        urls.append(_FULLURL)
        names.append(soup.select('a')[i].attrs['href'])

names_urls = zip(names, urls)

for name, url in names_urls:
    print(url)
    # rq = urllib3.Request(url)
    # res = urllib3.urlopen(rq)
    res = requests.get(url)
    name = re.sub("/","_",name)
    pdf = open(baseDir + "tmp/" + name, 'wb')
    pdf.write(res.content)
    pdf.close()
