![Open Terminal Logo](asset/ot_logo.png)
## Open Terminal
A "google-search" for stocks. Using [yfinance](https://github.com/ranaroussi/yfinance) under the hood.

deployed using [streamlit share](https://www.streamlit.io/sharing): [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/ohjho/open_terminal)

[![Open Terminal Screenshoot](asset/ot_screenshot.png)](https://share.streamlit.io/ohjho/open_terminal)

### Requirements

#### ~[ta-lib]((https://mrjbq7.github.io/ta-lib/)~
installing ta-lib is [non-trivial](https://github.com/mrjbq7/ta-lib/issues/127) especially on streamlit share. Fortunately, there's a [hack](https://github.com/mrjbq7/ta-lib/issues/127).

> Because of the complexity surrounding ta-lib, this project no longer requires it
> most of the TA calculuatioa can actually be performed in pandas/ numpy
