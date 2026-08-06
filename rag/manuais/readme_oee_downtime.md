**Dataset Title: OEE and Downtime Dataset from Heavy Clay Manufacturing Line (Version 1.0)**

# 1. Overview

This dataset contains operational and maintenance data from a continuous brick manufacturing production line. The data were collected as part of a Lean Six Sigma and Predictive Maintenance (PdM) research project aimed at improving line-level Overall Equipment Effectiveness (OEE). The production system operates as a serial line, where OEE losses in individual machines affect the entire production throughput.

# 2. Objectives:

The dataset supports the following analyses:

- OEE variability analysis

- Downtime and failure mode analysis

- Mean Time to Repair (MTTR) and Mean Time between Failures (MTBF) evaluation

- Identification of critical failures

- Identification of critical locations in production line

# 3. Industrial Context

- Industry: Heavy clay / brick manufacturing 

- Production type: Continuous production system 

- System characteristic: Serial production line with minimal buffers

- Data

# 4. File Description

## 4.1 OEE_dataset.csv

Summary: Line-level Overall Equipment Effectiveness data before Lean Six Sigma implementation


| **Column Name**              | **Description**                                         | **Unit**                   |
| ---------------------------- | ------------------------------------------------------- | -------------------------- |
| shiftDate                    | Date of specific shift                                  | MM-DD-YYYY                 |
| OEE                          | Overall Equipment Effectiveness of production line      | Closed Unit Interval [0,1] |
| Availability                 | Availability of production line                         | Closed Unit Interval [0,1] |
| Performance                  | Performance of production line                          | Closed Unit Interval [0,1] |
| Quality                      | Quality of production line                              | Closed Unit Interval [0,1] |
| SumofgoodProductionTimeSec   | Total time at which production line runs at full speed  | Time (Seconds)             |
| SumofslowProductionTimeSec   | Total time at which production line runs at lower speed | Time (Seconds)             |
| Sumof plannedTime            | Total planned production time                           | Time (Seconds)             |
| SumofidealProductionQuantity | Total ideal production time                             | Time (Seconds)             |
| Sum of plannedStopsTimeSec   | Total planned downtime                                  | Time (Seconds)             |
| Sum of unplannedStopsTimeSec | Total unplanned downtime                                | Time (Seconds)             |
| Sum of goodQty               | Total production items excluding defects                | Items                      |
| Sum of totalQty              | Total production items                                  | Items                      |

## 4.2 OEE_dataset_afterLSS

Summary: Line-level Overall Equipment Effectiveness data after Lean Six Sigma implementation


| **Column Name**              | **Description**                                         | **Unit**                   |
| ---------------------------- | ------------------------------------------------------- | -------------------------- |
| shiftDate                    | Date of specific shift                                  | MM-DD-YYYY                 |
| OEE                          | Overall Equipment Effectiveness of production line      | Closed Unit Interval [0,1] |
| Availability                 | Availability of production line                         | Closed Unit Interval [0,1] |
| Performance                  | Performance of production line                          | Closed Unit Interval [0,1] |
| Quality                      | Quality of production line                              | Closed Unit Interval [0,1] |
| SumofgoodProductionTimeSec   | Total time at which production line runs at full speed  | Time (Seconds)             |
| SumofslowProductionTimeSec   | Total time at which production line runs at lower speed | Time (Seconds)             |
| Sumof plannedTime            | Total planned production time                           | Time (Seconds)             |
| SumofidealProductionQuantity | Total ideal production time                             | Time (Seconds)             |
| Sum of plannedStopsTimeSec   | Total planned downtime                                  | Time (Seconds)             |
| Sum of unplannedStopsTimeSec | Total unplanned downtime                                | Time (Seconds)             |
| Sum of goodQty               | Total production items excluding defects                | Items                      |
| Sum of totalQty              | Total production items                                  | Items                      |

## 4.3 DowntimeDataset.csv

Event-based downtime records before Lean Six Sigma implementation

|   |   |   |
|---|---|---|
|**Column Name**|**Description**|**Unit**|
|Date|Date of specific shift|MM-DD-YYYY|
|Productcode|Product specific code|Nominal data|
|StopGroup|Downtime category (operational, failure, materials, setups-changeovers)|Nominal data|
|Stop|Performance of production line|Nominal data|
|StopType|Planned included in OEE/ Planned not included in OEE/ Unplanned|Nominal data|
|StopLocation|Downtime inducing production line location/machine|Nominal data|
|ExtraText|Operators note (kept in Greek, original language, to avoid translation mistakes)|Nominal data|
|StopStartTime|Total planned production time|YYYY-MM-DD HH:MM|
|StopEndTime|Total ideal production time|YYYY-MM-DD HH:MM|
|StopDuration(min)|Total planned downtime|Time (Minutes)|

## 4.4 DowntimeDataset_afterLSS.csv

Event-based downtime records after Lean Six Sigma implementation

|   |   |   |
|---|---|---|
|**Column Name**|**Description**|**Unit**|
|Date|Date of specific shift|MM-DD-YYYY|
|Productcode|Product specific code|Nominal data|
|StopGroup|Downtime category (operational, failure, materials, setups-changeovers)|Nominal data|
|Stop|Performance of production line|Nominal data|
|StopType|Planned included in OEE/ Planned not included in OEE/ Unplanned|Nominal data|
|StopLocation|Downtime inducing production line location/machine|Nominal data|
|ExtraText|Operators note (kept in Greek, original language, to avoid translation mistakes)|Nominal data|
|StopStartTime|Total planned production time|YYYY-MM-DD HH:MM|
|StopEndTime|Total ideal production time|YYYY-MM-DD HH:MM|
|StopDuration(min)|Total planned downtime|Time (Minutes)|

# 5. Data Collection

Manufacturing Execution System (MES): Evocon

Maintenance logs: Evocon

 OEE data: shift-based

 Downtime data: event-based

# 6. Data Preprocessing

The following preprocessing steps were applied:

- Check for duplicate records 

- Check for missing values

- Standardization of timestamps  and categorical/nominal data

# 7. Methodological Notes

Data derived from serial production line, where machine specific OEE losses propagate to system-level.

- Preventive maintenance is time-based and occurs during planned shutdown periods.

- Downtime classification follows internal maintenance categorization.

- Line-OEE is calculated using production line constraint (bottleneck) for availability and performance, as well as total defects:

        _Line-OEE = Availability (constraint) × Performance (constraint) × Quality (total)_

# 8. Usage Notes

This dataset is suitable for:

- Statistical analysis

- Reliability analysis (MTBF, MTTR)

- Process improvement studies

Users are advised to validate assumptions before applying models.

# 9. Licensing

This dataset is licensed under:

Creative Commons Attribution 4.0 International (CC BY 4.0)

#  12. Funding

This dataset was generated as part of a PhD in Industry 1123/0108 project, funded by the Research and Innovation Foundation under the RESTART 2016–2020 Programme for Research, Technological Development and Innovation.

# 13. Contact

For questions or collaborations:

Contact person: Panos Ntoas

Email: [pntoas01@ucy.ac.cy](mailto:pntoas01@ucy.ac.cy) / [panos.doas@kakoyiannisbricks.com](mailto:panos.doas@kakoyiannisbricks.com)

Affiliation: University of Cyprus / Kakoyiannis Bricks Ltd

# 14. Provenance and Data Collection Period

DOI: 10.5281/zenodo.17855209 (published on Zenodo, December 8, 2025, Version 1.0).

Data collection period: from 2025-09-22 (production start) to 2025-12-18 (production end).

Supervisor: Andreas Kyprianou. Hosting institution: University of Cyprus. This dataset is
part of a PhD thesis in Mechanical and Manufacturing Engineering (University of Cyprus).