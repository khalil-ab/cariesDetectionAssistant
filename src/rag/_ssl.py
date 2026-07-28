"""
Contournement de l'interception SSL (proxy MITM) pour les appels API en local.

Certains reseaux (entreprise, antivirus) remplacent les certificats TLS, ce qui
fait echouer la verification. Si la variable d'environnement CARIES_INSECURE_SSL=1
est definie, on desactive la verification des certificats.

A n'utiliser qu'en local derriere un tel proxy. En ligne (Colab, serveur),
laisser la variable absente : la verification reste active.
"""

import os


def maybe_disable_ssl():
    if os.environ.get("CARIES_INSECURE_SSL") != "1":
        return
    import ssl
    import requests
    import urllib3

    urllib3.disable_warnings()
    ssl._create_default_https_context = ssl._create_unverified_context

    _origine = requests.Session.merge_environment_settings

    def _sans_verif(self, url, proxies, stream, verify, cert):
        parametres = _origine(self, url, proxies, stream, verify, cert)
        parametres["verify"] = False
        return parametres

    requests.Session.merge_environment_settings = _sans_verif
