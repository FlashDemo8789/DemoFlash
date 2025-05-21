# FLASH Metrics Guide

## Introduction

This guide documents the metrics system in FLASH, including the metrics cleanup process, naming conventions, and best practices for maintaining data quality.

## Metrics Structure

Metrics in FLASH are organized by pillars:

- **Advantage**: Metrics related to competitive advantage and differentiation
- **Market**: Metrics related to market size, growth, and user engagement
- **Capital**: Metrics related to funding, revenue, and financial health
- **People**: Metrics related to team size, experience, and diversity
- **Info**: Contextual information not used directly in models

## Canonical Metric Names

We maintain a standardized set of canonical metric names. Each metric has one official name, as shown in the table below.

| Pillar             | Canonical Name                | Previous Aliases                       | Description                                     |
| ------------------ | ----------------------------- | -------------------------------------- | ----------------------------------------------- |
| **Advantage**      | `patent_count`                | `patents_count`                        | Number of patents held by the company           |
|                    | `has_network_effect`          | `network_effects_present`              | Whether the product has network effects (bool)  |
| **Market**         | `nps_score`                   | `nps`                                  | Net Promoter Score (0-100)                      |
|                    | `burn_rate_usd`               | `monthly_burn_usd`                     | Monthly cash burn rate in USD                   |
| **Capital**        | `total_funding_usd`           | `total_capital_raised_usd`             | Total funding raised in USD                     |
|                    | `revenue_annual_usd`          | `annual_revenue_run_rate`              | Annual revenue in USD                           |
| **People**         | `founders_count`              | `founding_team_size`                   | Number of founders                              |
|                    | `domain_expertise_years_avg`  | `founder_domain_experience_years`      | Average years of domain expertise               |
|                    | `previous_exits_count`        | `prior_successful_exits_count`         | Number of previous successful exits             |
| **Info / Context** | `sector`                      | `industry`                             | Industry sector classification                  |

## Metrics Deduplication Process

The metrics cleanup process eliminates duplicate metrics in the following ways:

1. **Identifying duplicates**: We identify metrics that represent the same concept but have different names.
2. **Mapping to canonical names**: We establish a canonical name for each metric and map aliases to it.
3. **Updating configuration files**: We update `metrics.json` files in both frontend and backend to use only canonical names.
4. **Cleaning datasets**: We transform datasets to use only canonical names, dropping duplicates.
5. **Testing for collinearity**: We run tests to ensure no columns are highly correlated (which would indicate duplication).

## Implementation

The metrics cleanup implementation consists of:

- `scripts/cleanup_duplicate_metrics.py`: Main script to deduplicate metrics
- `scripts/test_metrics_collinearity.py`: Unit tests to verify cleanup and catch future duplicates
- `scripts/verify_metrics_cleanup.py`: Script to verify that duplicate metrics have been removed from all parts of the system
- `scripts/cleanup_and_verify_metrics.py`: Combined script that runs cleanup, updates schemas, and verifies the changes
- GitHub workflow: CI checks to prevent duplicate metrics from being reintroduced

### Using the Cleanup Scripts

#### Basic Cleanup

To clean up duplicate metrics in your local environment:

```bash
python scripts/cleanup_duplicate_metrics.py
```

This will:
1. Read the dataset with duplicate metrics
2. Map duplicate columns to their canonical names
3. Write a clean version of the dataset
4. Update metrics JSON files to remove duplicate definitions
5. Test for any remaining collinearity

#### Verification

To verify that duplicate metrics have been removed from all parts of the system:

```bash
python scripts/verify_metrics_cleanup.py
```

This script checks:
1. The cleaned CSV file doesn't contain duplicate metrics
2. The metrics.json files in both frontend and backend are clean
3. The Pydantic schema in backend/app/schemas.py is clean
4. The TypeScript types in frontend/types/metrics.ts are clean

#### Combined Cleanup and Verification

For convenience, you can run the combined script that performs cleanup, updates schemas, and verifies the changes:

```bash
python scripts/cleanup_and_verify_metrics.py
```

This script:
1. Runs cleanup_duplicate_metrics.py to clean up duplicate metrics
2. Runs gen_metrics.py to update schema files
3. Runs verify_metrics_cleanup.py to verify the cleanup

## Best Practices for Adding New Metrics

When adding new metrics to the system:

1. **Check existing metrics**: Before adding a new metric, check if a similar metric already exists.
2. **Follow naming conventions**: Use consistent naming (`snake_case`, suffixes indicating units like `_usd`, `_percent`).
3. **Document in metrics.json**: Add the metric with appropriate metadata (label, type, pillar, etc.).
4. **Update schemas**: Ensure any type definitions are updated for frontend and backend.
5. **Run collinearity tests**: Verify the new metric doesn't duplicate an existing one.

## Troubleshooting

If you encounter issues with metrics:

- **Duplicate names**: Run the cleanup script to standardize metrics.
- **Collinearity warnings**: Investigate columns with high correlation to identify potential duplication.
- **Missing metrics**: Check if the metric was renamed as part of the deduplication process.

## Metric Schema Reference

Each metric in the JSON configuration includes:

```json
{
  "key": "metric_name",         // Machine-readable identifier
  "label": "Human Label",       // User-facing label
  "type": "number|text|checkbox|list", // Data type
  "default": 0,                 // Default value
  "pillar": "Category"          // Organizational category
}
``` 