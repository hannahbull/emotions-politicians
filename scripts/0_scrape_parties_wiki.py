import pandas as pd
import re
import wikipedia
import unicodedata

elections = pd.read_csv("../data/parties/view_election.csv", encoding="Latin1")

electemp = elections.groupby('country_name_short')['election_date'].max().reset_index(name='max_date')
electemp = electemp.drop([2])
elections = elections.merge(electemp, on='country_name_short')
elections = elections[elections.election_date == elections.max_date]

elections['leader'] = ''
elections['searchterm'] = ''

for i in range(1, len(elections)):
    try:
        requestph = elections['party_name_english'].iloc[i] + ' ' + '(' + elections['country_name'].iloc[i] + ')'
        print(requestph)
        requestph = wikipedia.search(requestph)
        requestphrase = requestph[0]
        print(requestphrase)
        elections['searchterm'].iloc[i] = requestphrase

        wikipage = wikipedia.WikipediaPage(requestphrase).html()

        all_chars = (chr(i) for i in range(0x110000))
        control_chars = ''.join(c for c in all_chars if unicodedata.category(c) == 'Cc')

        control_char_re = re.compile('[%s]' % re.escape(control_chars))


        def remove_control_chars(s):
            return control_char_re.sub('', s)


        wikipage = remove_control_chars(wikipage)
        wikipage = wikipage[wikipage.find("infobox"):wikipage.find("Navigation menu")]
        wikipage = re.sub("<.*?>", " ", wikipage)
        firstword = (wikipage.find("Leader"), wikipage.find("leader"),
                     wikipage.find("President"), wikipage.find("president"), wikipage.find("Chair"))

        firstword = min(i for i in firstword if i > 0)
        wikipage = wikipage[firstword:]
        wikipage = re.sub("^[ ]*", "", wikipage)
        wikipage = re.sub("^[^  ]+[ ]{2}", "", wikipage)
        wikipage = re.sub("[ ]{2}.*$", "", wikipage)
        wikipage = re.sub("^[ ]*", "", wikipage)
        wikipage = re.sub("[ ]*$", "", wikipage)
        elections['leader'].iloc[i] = wikipage
        print(wikipage)

        print(i)
    except:
        print("warning")

print(elections)
elections.to_csv("../data/parties/added_parties.csv")