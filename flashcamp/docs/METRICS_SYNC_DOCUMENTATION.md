# Flash DNA Metrics Synchronization Technical Documentation

**Version:** 1.0  
**Date:** May 3, 2025  
**Author:** Flash DNA Engineering Team

## Overview

This document provides technical documentation for the metrics synchronization system implemented in the Flash DNA platform. The system ensures perfect consistency of metrics definitions across the entire application stack:

- CSV Data Files (Source of Truth)
- Backend Models and Constants
- Frontend TypeScript Definitions
- UI Components

## Table of Contents

1. [Metrics Synchronization Architecture](#metrics-synchronization-architecture)
2. [Core Components](#core-components)
3. [Key Files and Their Roles](#key-files-and-their-roles)
4. [Synchronization Process](#synchronization-process)
5. [Developer Guide](#developer-guide)
6. [Troubleshooting](#troubleshooting)
7. [Future Enhancements](#future-enhancements)

---

## Metrics Synchronization Architecture

The metrics synchronization system follows a unidirectional flow architecture:

```
CSV Data (Source of Truth) → Backend Constants → TypeScript Types → Frontend Components
```

### Key Principles

1. **Single Source of Truth**: The CSV file (`camp_plus_balanced_with_meta.csv`) serves as the definitive source of truth for all metrics.
2. **Automated Synchronization**: The `sync_metrics.py` tool ensures all system components stay in sync.
3. **Type Safety**: Generated TypeScript types provide compile-time safety for frontend code.
4. **Consistent Interfaces**: Backend models mirror frontend types to ensure API consistency.

---

## Core Components

### 1. Metrics Synchronization Tool

The `sync_metrics.py` script is the central component responsible for maintaining consistency. It:

- Extracts headers from the CSV file (source of truth)
- Updates frontend metrics.json with any missing metrics
- Generates TypeScript type definitions for frontend components
- Updates Python backend models with synchronized metrics
- Validates usage of metrics across key components
- Generates usage reports for analytics

### 2. Metrics Type System

The system implements a comprehensive type system across all layers:

- **Backend**: Python types via Pydantic models
- **Frontend**: TypeScript interfaces and enums
- **API**: JSON schema for API documentation

### 3. Component Integration

All UI components have been updated to use the type system, ensuring:

- Consistent prop types
- Proper handling of all 99 metrics
- Type-safe access to metric properties
- Consistent UI rendering

---

## Key Files and Their Roles

### Core Files

| File | Purpose |
|------|---------|
| `/flashcamp/data/camp_plus_balanced_with_meta.csv` | Source of truth containing all 99 metrics as CSV headers |
| `/flashcamp/sync_metrics.py` | Synchronization tool that maintains consistency across all components |
| `/flashcamp/constants/metrics.py` | Backend Python constants defining all metrics |
| `/flashcamp/frontend/src/constants/metrics.json` | Frontend JSON definitions of all metrics |
| `/flashcamp/frontend/types/metrics.ts` | Generated TypeScript types for frontend components |
| `/flashcamp/backend/schemas.py` | Pydantic models for the backend API |
| `/flashcamp/metrics_usage_report.md` | Generated report showing which components use which metrics |

### Frontend Components

| Component | Purpose |
|-----------|---------|
| `MetricInput.tsx` | Renders individual metrics with appropriate input controls |
| `PillarStep.tsx` | Groups metrics by pillar in the UI |
| `WizardPage.tsx` | Main form interface for entering metrics data |
| `ResultsPage.tsx` | Displays analysis results based on metrics |
| `PDFDownloadButton.tsx` | Generates reports containing all metrics |
| `RadarView.tsx` | Visualizes pillar scores based on metrics |

---

## Synchronization Process

### Initial Sync

The initial synchronization process:

1. Extracts headers from the CSV file to determine all required metrics
2. Compares with frontend metrics.json and adds any missing metrics
3. Updates backend constants to match frontend definitions
4. Generates TypeScript types for the frontend
5. Updates Pydantic models in the backend
6. Validates all components to ensure they can handle all metrics

### Ongoing Maintenance

When you add new metrics to the system:

1. Add the new metric to the CSV file first
2. Run `sync_metrics.py` to propagate changes through the system
3. Verify changes through the generated usage report

### Type Generation

The type generation process creates:

1. **TypeScript enums** for metric names, pillars, and types
2. **TypeScript interfaces** for individual metrics and metric values
3. **Helper types** for grouping metrics by pillar

---

## Developer Guide

### Adding a New Metric

To add a new metric to the system:

1. Add the metric as a new column in `/flashcamp/data/camp_plus_balanced_with_meta.csv`
2. Run the sync tool:
   ```bash
   cd /Users/sf/Desktop/FLASH/flashcamp
   python sync_metrics.py
   ```
3. The script will:
   - Add the metric to `metrics.json` with appropriate defaults
   - Update TypeScript types
   - Update backend models
4. Review and update any component-specific logic that might need to handle the new metric differently

### Type-Safe Component Development

When developing components that use metrics:

1. Import types from `/frontend/types/metrics.ts`:
   ```typescript
   import { Metric, MetricValues, MetricName, MetricPillar } from "../types/metrics";
   ```

2. Use type-safe access to metric properties:
   ```typescript
   // Type-safe way to access a metric value
   const value = form[metric.name as keyof FormState];
   ```

3. Handle all possible metric types:
   ```typescript
   switch (metric.type) {
     case "number":
       // Handle number input
       break;
     case "checkbox":
       // Handle boolean input
       break;
     case "text":
       // Handle text input
       break;
     case "list":
       // Handle list input
       break;
   }
   ```

### Troubleshooting TypeScript Imports

If you encounter TypeScript import issues:

1. Define local interfaces in the component file:
   ```typescript
   // Local metric interface
   interface Metric {
     name: string;
     label: string;
     pillar: string;
     type: string;
     // Add other needed properties
   }
   ```

2. Use a type assertion when needed:
   ```typescript
   const metricsJson: Metric[] = metricsJsonRaw as Metric[];
   ```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Missing metric in frontend | Run `python sync_metrics.py` to synchronize metrics |
| TypeScript errors about missing properties | Ensure you're using the correct metric types with optional properties |
| Backend API errors | Restart the backend after updating metrics to reload Pydantic models |
| Inconsistent UI rendering | Check the component's switch statement handles all metric types |

### Validation Errors

If you encounter validation errors in the backend:

1. Check if the metric exists in `constants/metrics.py`
2. Verify the type is consistent between frontend and backend
3. Run the sync tool to fix inconsistencies

---

## Future Enhancements

Planned enhancements to the metrics synchronization system:

1. **Automated testing** to verify metrics consistency on CI/CD pipelines
2. **Migration scripts** for handling breaking changes to metrics
3. **Versioning system** for tracking metrics changes over time
4. **UI for metrics management** to provide an admin interface for defining metrics

---

## Conclusion

The metrics synchronization system provides a robust foundation for maintaining consistency across the Flash DNA platform. By following the guidelines in this documentation, developers can confidently add, modify, and use metrics without worrying about inconsistencies between different parts of the system.

For questions or support, contact the Flash DNA Engineering Team.
