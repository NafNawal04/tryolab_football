from inference.colors import black, blue, green, navy_blue, sky_blue, white, red

chelsea_filter = {"name": "Chelsea", "colors": [blue, green]}

city_filter = {"name": "Man City", "colors": [sky_blue]}

real_madrid_filter = {"name": "Real Madrid", "colors": [white]}
barcelona_filter = {"name": "Barcelona", "colors": [navy_blue, red]}

france_filter = {"name": "France", "colors": [navy_blue, white]}
croatia_filter = {"name": "Croatia", "colors": [red, white]}

referee_filter = {"name": "Referee", "colors": [black]}

DEFAULT_MATCH_KEY = "chelsea_man_city"

_match_filters = {
    "chelsea_man_city": [chelsea_filter, city_filter, referee_filter],
    "real_madrid_barcelona": [real_madrid_filter, barcelona_filter, referee_filter],
    "france_croatia": [france_filter, croatia_filter, referee_filter],
}


def get_filters_for_match(match_key: str):
    try:
        return _match_filters[match_key]
    except KeyError as exc:
        raise ValueError(f"Unknown match key '{match_key}'") from exc


filters = _match_filters[DEFAULT_MATCH_KEY]
