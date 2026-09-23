"""certify — exact certification of the minimal recurrent behavioral memory of an enumerable environment.

    from certify import Adapter, certify
    cert = certify(my_adapter); print(cert.table())

Command line: python -m certify --task memchain --L 30 --bits 2      (see README.md)
"""
from .core import Adapter, Certificate, certify, certify_env, record

__all__ = ["Adapter", "Certificate", "certify", "certify_env", "record"]
