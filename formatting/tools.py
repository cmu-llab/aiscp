import panphon

def preprocess_phone(ft, phone):
    if not ft:
        ft = panphon.FeatureTable()

    STOP = {
        'son': -1,
        'cont': -1
    }
    FRICATIVE = {
        'son': -1,
        'cont': 1
    }
    if phone[0] == "*":
        phone = phone[1:]
    elif len(phone) >= 2 and \
        ft.fts(phone[0]) and ft.fts(phone[0]).match(STOP) and ft.fts(phone[1]) and ft.fts(phone[1]).match(FRICATIVE):
        # add ligature to affricates (stop + fricative)
        phone = phone[0] + '͡' + phone[1]

    return phone.strip()
