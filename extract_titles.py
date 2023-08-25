from bs4 import BeautifulSoup
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

# HTML from https://www.imdb.com/chart/top/ goes here
html_data = """
"""

soup = BeautifulSoup(html_data, 'html.parser')
titles = soup.find_all('h3', class_='ipc-title__text')

print('film_titles = [\n    ',end='')
for title in titles:
    clean_title = re.sub(r'^\d+\.\s', '', title.text)
    print(f"\"{clean_title}\", ",end='')
print('\n]')
