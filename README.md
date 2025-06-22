## Data module
The code has a data module that fetches it directly from FRED.
It's (relatively) simple to use. You'll need to create a yaml file with the following fields:

```
api_key: < Your FRED API key >
real_time: <True if you want first release data, default=False>
sample_start: <string 'MM-DD-YYYY>

data_series: # field with your data series
  name_of_series:
    label: <string>
    code: <FRED code>
    transformation: <Options: 'yoy-m', 'yoy-q', 'log', 'linear'>
    interpolate: <Interpolation if you want. You should pass the pandas interpolation argument>
    revision: <'get_rate' or blank. Use this in case you have data that will be revised>
    accumulate: <'period-over-period' if you want accumulation>
```
