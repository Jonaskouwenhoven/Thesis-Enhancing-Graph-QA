"""
    This module contains functions that are used by all other modules in this project.
"""
import re
import requests
import time
from json import JSONDecodeError
from requests import RequestException
from typing import Union, Callable
from urllib.parse import urlparse
from urllib.error import URLError

TRUSTED_SOURCES = ['localhost', 'cbs.nl', 'cbscms9', 'rivm.nl', 'politie.nl', 'grensdata.eu', 'volkstellingen.nl']


class TooManyRequestsError(Exception):
    """Exception when to many requests are received by application."""
    pass


def secure_request(callee: Union[Callable, str], *args, json=True, max_retries=3, verify=True, timeout=1):
    """
        Do a get request by an url or function and do 3 retries in case of non-critical failures.
        Only use this method for background/crawling activities and not for requests in real-time.

        :param callee: url or function to do get request on
        :param args: optional arguments to pass to callee 'getter' function
        :param json: return json or content from request result (default: json)
        :param max_retries: maximum number of tries if request fails before returning None (default: 3)
        :param verify: do request with SSL verification (default: True)
        :param timeout: the allowed number of seconds timeout
        :return: request content or json from request source. None if no data found. False if failed
    """
    for _ in range(max_retries):
        try:
            if callable(callee):
                req = callee(*args, timeout=(timeout, timeout), verify=verify)
            elif isinstance(callee, str):
                url = urlparse(callee)
                if re.search(fr"({'|'.join(TRUSTED_SOURCES)})$", url.hostname, re.IGNORECASE) is None:
                    raise URLError(f"{callee} is not in the list of trusted sources {TRUSTED_SOURCES}")

                req = requests.get(callee, timeout=(timeout, timeout), verify=verify)
            else:
                raise RequestException("Callee is not a valid callable or retrievable URL")

            if req.status_code == 429:
                raise TooManyRequestsError
            if req.status_code != 200:
                print(f"Fetching {callee} succeeded, but yielded status code {req.status_code}")
                return None

            data = req.json() if json else req.content
        except (TimeoutError, TooManyRequestsError):
            # Try to fetch again if timed out
            print(f"Request timed out or too many request for {callee}.")
            time.sleep(3)
            continue
        except JSONDecodeError:
            print(f"Could not parse JSON-result for request {callee}")
            break
        except RequestException as e:
            print(f"Failed to fetch {callee}: {str(e)}")
            break
        else:
            if callable(callee):
                return req
            return data

    return False
