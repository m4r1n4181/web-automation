# CSS selectors for the web application
SELECTORS = {
    'login': {
        # Selectors for the login page
        'text': 'input[name="userid"]',
        'password': 'input[name="password"]',
        'submit': 'input[type="submit"]'
    },
    'table': {
        # Selectors for navigating and identifying elements within the payment table
        'first_empty_row': 'table tbody tr:last-child',
        'name_input': 'input[name="kupacNaziv"]',
        'amount_input': 'input[name="iznos"]',
    },
    'menu': {
        # Selectors for navigation menus and specific items
        'glavni': '#kpmenu b',
        'stavka_3': '#ulmenu > li:nth-child(3) > a',
        'podstavka_16': '#ul_3 > li:nth-child(15) > a',
        'tabela_celija': '#hrow_1 td.r:nth-child(9)',
        'stavka_5': '#ulmenu > li:nth-child(5) > a',
        'podstavka_5': '#ul_5 > li:nth-child(5) > a',
    },
    'actions': {
        # Selectors for action buttons
        'save': '#save_butt'
    }
}