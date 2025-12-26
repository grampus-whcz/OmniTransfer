
## How to get the no fault time zone for each fault in the groud truth
In multivariate time series anomaly detection, it is essential to compare anomalous (faulty) segments with normal (non-faulty) counterparts under comparable conditions. To facilitate such analysis, for each fault event recorded in the ground-truth file (`record.csv`), we identify a corresponding 30-minute non-fault interval from another day based on the following matching strategy:

1. **Same-time preference**: If the same half-hour window (e.g., 09:00–09:30) on the other day contains no faults, it is selected as the normal reference interval.
2. **Nearest-neighbor fallback**: If the corresponding window on the other day also contains faults, the nearest adjacent 30-minute window (in chronological order) that is free of faults on that day is chosen instead.

For every fault entry, the program outputs an enhanced CSV file that preserves all original columns and appends four new attributes describing the matched normal interval:
- `normal_start_timestamp`: Unix timestamp (seconds) of the normal interval start,
- `normal_start_time`: Human-readable start time in the format `YYYY-MM-DD HH:MM:SS`,
- `normal_end_timestamp`: Unix timestamp of the normal interval end,
- `normal_end_time`: Human-readable end time in the format `YYYY-MM-DD HH:MM:SS`.

This alignment ensures that anomalous behaviors can be rigorously contrasted against semantically similar, healthy system states—enabling more accurate and context-aware multivariate anomaly detection.

## cloudbed-1 fault pattern
```
Count    Fault Pattern (level | reason)
------------------------------------------------------------
5        pod | container read I/O load
5        node | node memory consumption
5        node | node disk read I/O consumption
5        node | node disk write I/O consumption
5        node | node disk space consumption
4        service | container CPU load
4        node | node CPU load
4        service | container memory load
3        service | container network latency
3        pod | container memory load
3        service | container network packet corruption
3        service | container process termination
3        pod | container network packet retransmission
3        pod | container network packet corruption
3        pod | container CPU load
3        service | container read I/O load
2        node | node CPU spike
2        service | container packet loss
2        service | container network packet retransmission
1        pod | container process termination
1        pod | container network latency
1        service | container write I/O load

```

## cloudbed-2 fault pattern
'*' represents the fault pattern exists alone in cloudbed-2
```
Count    Fault Pattern (level | reason)
------------------------------------------------------------
5        node | node CPU load
5        pod | container network packet retransmission
5        service | container read I/O load
5        node | node disk space consumption
5        node | node disk write I/O consumption
4        pod | container memory load
4        service | container network packet corruption
4        pod | container read I/O load
4        node | node disk read I/O consumption
4        pod | container CPU load
4        service | container packet loss
4        node | node memory consumption
3        pod | container write I/O load  *
3        service | container network packet retransmission
3        pod | container network packet corruption
2        service | container network latency
2        node | node CPU spike
2        pod | container network latency
2        pod | container process termination
2        service | container memory load
2        pod | container packet loss  *
2        service | container CPU load
1        service | container write I/O load
1        service | container process termination
```

## Fault information

```
cloudbed-1 Top 10 largest time intervals between consecutive faults:

From: 2022-03-21 16:41:47  →  To: 2022-03-21 20:26:07
  Interval: 13460.0 seconds (3.74 hours)
  Previous fault: node-5 (node disk space consumption)
  Current fault:  node-3 (node disk space consumption)
--------------------------------------------------------------------------------
From: 2022-03-21 05:36:03  →  To: 2022-03-21 07:26:56
  Interval: 6653.0 seconds (1.85 hours)
  Previous fault: shippingservice2-0 (container network packet corruption)
  Current fault:  adservice2-0 (container read I/O load)
--------------------------------------------------------------------------------
From: 2022-03-21 20:26:07  →  To: 2022-03-21 21:52:20
  Interval: 5173.0 seconds (1.44 hours)
  Previous fault: node-3 (node disk space consumption)
  Current fault:  node-2 (node memory consumption)
--------------------------------------------------------------------------------
From: 2022-03-21 12:18:41  →  To: 2022-03-21 13:39:06
  Interval: 4825.0 seconds (1.34 hours)
  Previous fault: productcatalogservice (container network latency)
  Current fault:  currencyservice (container memory load)
--------------------------------------------------------------------------------
From: 2022-03-20 23:30:10  →  To: 2022-03-21 00:37:25
  Interval: 4035.0 seconds (1.12 hours)
  Previous fault: productcatalogservice (container CPU load)
  Current fault:  cartservice (container network packet retransmission)
--------------------------------------------------------------------------------
From: 2022-03-20 14:23:15  →  To: 2022-03-20 15:27:23
  Interval: 3848.0 seconds (1.07 hours)
  Previous fault: shippingservice (container CPU load)
  Current fault:  shippingservice (container process termination)
--------------------------------------------------------------------------------
From: 2022-03-21 22:13:02  →  To: 2022-03-21 23:15:29
  Interval: 3747.0 seconds (1.04 hours)
  Previous fault: node-5 (node disk write I/O consumption)
  Current fault:  node-4 (node disk space consumption)
--------------------------------------------------------------------------------
From: 2022-03-20 12:10:57  →  To: 2022-03-20 13:13:19
  Interval: 3742.0 seconds (1.04 hours)
  Previous fault: recommendationservice (container CPU load)
  Current fault:  node-2 (node disk write I/O consumption)
--------------------------------------------------------------------------------
From: 2022-03-21 10:44:52  →  To: 2022-03-21 11:40:30
  Interval: 3338.0 seconds (0.93 hours)
  Previous fault: node-6 (node memory consumption)
  Current fault:  frontend (container memory load)
--------------------------------------------------------------------------------
From: 2022-03-20 22:01:12  →  To: 2022-03-20 22:56:04
  Interval: 3292.0 seconds (0.91 hours)
  Previous fault: frontend-1 (container network latency)
  Current fault:  node-3 (node disk read I/O consumption)
--------------------------------------------------------------------------------

cloudbed-1 检测到 2 天的数据: [datetime.date(2022, 3, 20), datetime.date(2022, 3, 21)]

🔍 故障按每日半小时时段对齐对比（时间段格式：HH:MM - HH:MM）

====================================================================================================

🕒 00:30 - 01:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • cartservice | container network packet retransmission

🕒 01:00 - 01:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (2 fault(s)):
    • cartservice-0 | container network packet retransmission
    • productcatalogservice-2 | container read I/O load

🕒 01:30 - 02:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • node-5 | node disk read I/O consumption

🕒 02:00 - 02:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (2 fault(s)):
    • node-5 | node disk write I/O consumption
    • recommendationservice | container process termination

🕒 03:00 - 03:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • checkoutservice | container network packet corruption

🕒 03:30 - 04:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • node-6 | node disk write I/O consumption

🕒 04:30 - 05:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • cartservice | container network packet retransmission

🕒 05:00 - 05:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • node-4 | node memory consumption

🕒 05:30 - 06:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • shippingservice2-0 | container network packet corruption

🕒 07:00 - 07:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • adservice2-0 | container read I/O load

🕒 07:30 - 08:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (2 fault(s)):
    • node-6 | node disk read I/O consumption
    • productcatalogservice-0 | container CPU load

🕒 08:00 - 08:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • shippingservice-1 | container CPU load

🕒 08:30 - 09:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • adservice | container network latency
    • cartservice | container network latency
📅 2022-03-21 (1 fault(s)):
    • node-1 | node CPU load

🕒 09:00 - 09:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • shippingservice-1 | container read I/O load
📅 2022-03-21 (2 fault(s)):
    • productcatalogservice | container CPU load
    • checkoutservice | container memory load

🕒 09:30 - 10:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • emailservice-0 | container read I/O load
📅 2022-03-21 (1 fault(s)):
    • shippingservice | container read I/O load

🕒 10:00 - 10:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-1 | node memory consumption
📅 2022-03-21 (1 fault(s)):
    • node-6 | node CPU spike

🕒 10:30 - 11:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-1 | node memory consumption
📅 2022-03-21 (2 fault(s)):
    • node-4 | node disk space consumption
    • node-6 | node memory consumption

🕒 11:00 - 11:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • shippingservice-1 | container memory load
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 11:30 - 12:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • recommendationservice | container network packet corruption
📅 2022-03-21 (1 fault(s)):
    • frontend | container memory load

🕒 12:00 - 12:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • node-1 | node disk read I/O consumption
    • recommendationservice | container CPU load
📅 2022-03-21 (1 fault(s)):
    • productcatalogservice | container network latency

🕒 13:00 - 13:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-2 | node disk write I/O consumption
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 13:30 - 14:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • node-4 | node CPU load
    • node-6 | node CPU load
📅 2022-03-21 (1 fault(s)):
    • currencyservice | container memory load

🕒 14:00 - 14:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • shippingservice | container CPU load
    • recommendationservice2-0 | container process termination
📅 2022-03-21 (2 fault(s)):
    • emailservice | container read I/O load
    • emailservice | container write I/O load

🕒 14:30 - 15:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • adservice | container read I/O load

🕒 15:00 - 15:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • shippingservice | container process termination
📅 2022-03-21 (1 fault(s)):
    • frontend2-0 | container network packet corruption

🕒 15:30 - 16:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • cartservice | container packet loss

🕒 16:00 - 16:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • shippingservice-1 | container network packet retransmission
📅 2022-03-21 (1 fault(s)):
    • frontend-2 | container read I/O load

🕒 16:30 - 17:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • currencyservice-0 | container network packet corruption
📅 2022-03-21 (1 fault(s)):
    • node-5 | node disk space consumption

🕒 17:00 - 17:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • node-4 | node CPU spike
    • paymentservice | container process termination
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 17:30 - 18:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-5 | node disk write I/O consumption
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 18:00 - 18:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • paymentservice | container memory load
    • emailservice | container network packet corruption
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 19:00 - 19:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • frontend-2 | container network packet retransmission
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 19:30 - 20:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-4 | node disk read I/O consumption
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 20:00 - 20:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • shippingservice2-0 | container CPU load
📅 2022-03-21 (1 fault(s)):
    • node-3 | node disk space consumption

🕒 21:00 - 21:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • emailservice | container packet loss
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 21:30 - 22:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • cartservice-0 | container memory load
    • node-1 | node disk space consumption
📅 2022-03-21 (1 fault(s)):
    • node-2 | node memory consumption

🕒 22:00 - 22:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • frontend-1 | container network latency
📅 2022-03-21 (1 fault(s)):
    • node-5 | node disk write I/O consumption

🕒 22:30 - 23:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-3 | node disk read I/O consumption
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 23:00 - 23:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • checkoutservice-2 | container memory load
📅 2022-03-21 (1 fault(s)):
    • node-4 | node disk space consumption

🕒 23:30 - 00:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • productcatalogservice | container CPU load
📅 2022-03-21 (1 fault(s)):
    • node-4 | node CPU load

✅ 对齐分析完成。


cloudbed-2 Top 10 largest time intervals between consecutive faults:

From: 2022-03-21 15:19:45  →  To: 2022-03-21 16:46:50
  Interval: 5225.0 seconds (1.45 hours)
  Previous fault: paymentservice (container process termination)
  Current fault:  node-1 (node disk space consumption)
--------------------------------------------------------------------------------
From: 2022-03-21 16:46:50  →  To: 2022-03-21 18:08:02
  Interval: 4872.0 seconds (1.35 hours)
  Previous fault: node-1 (node disk space consumption)
  Current fault:  checkoutservice (container network packet corruption)
--------------------------------------------------------------------------------
From: 2022-03-20 19:42:12  →  To: 2022-03-20 20:58:52
  Interval: 4600.0 seconds (1.28 hours)
  Previous fault: recommendationservice-2 (container memory load)
  Current fault:  productcatalogservice (container network packet retransmission)
--------------------------------------------------------------------------------
From: 2022-03-21 05:22:50  →  To: 2022-03-21 06:20:29
  Interval: 3459.0 seconds (0.96 hours)
  Previous fault: productcatalogservice-0 (container write I/O load)
  Current fault:  node-4 (node memory consumption)
--------------------------------------------------------------------------------
From: 2022-03-20 16:50:36  →  To: 2022-03-20 17:47:45
  Interval: 3429.0 seconds (0.95 hours)
  Previous fault: cartservice-0 (container network latency)
  Current fault:  adservice2-0 (container network latency)
--------------------------------------------------------------------------------
From: 2022-03-20 09:37:49  →  To: 2022-03-20 10:33:54
  Interval: 3365.0 seconds (0.93 hours)
  Previous fault: productcatalogservice (container network packet corruption)
  Current fault:  currencyservice-0 (container read I/O load)
--------------------------------------------------------------------------------
From: 2022-03-20 17:47:45  →  To: 2022-03-20 18:41:47
  Interval: 3242.0 seconds (0.90 hours)
  Previous fault: adservice2-0 (container network latency)
  Current fault:  node-2 (node disk space consumption)
--------------------------------------------------------------------------------
From: 2022-03-21 21:51:52  →  To: 2022-03-21 22:45:06
  Interval: 3194.0 seconds (0.89 hours)
  Previous fault: node-5 (node memory consumption)
  Current fault:  frontend (container network packet corruption)
--------------------------------------------------------------------------------
From: 2022-03-21 02:32:15  →  To: 2022-03-21 03:23:09
  Interval: 3054.0 seconds (0.85 hours)
  Previous fault: cartservice (container network packet corruption)
  Current fault:  node-5 (node disk space consumption)
--------------------------------------------------------------------------------
From: 2022-03-21 04:33:47  →  To: 2022-03-21 05:22:50
  Interval: 2943.0 seconds (0.82 hours)
  Previous fault: cartservice (container packet loss)
  Current fault:  productcatalogservice-0 (container write I/O load)
--------------------------------------------------------------------------------



cloudbed-2 检测到 2 天的数据: [datetime.date(2022, 3, 20), datetime.date(2022, 3, 21)]

🔍 故障按每日半小时时段对齐对比（时间段格式：HH:MM - HH:MM）

====================================================================================================

🕒 00:00 - 00:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (2 fault(s)):
    • node-2 | node disk read I/O consumption
    • shippingservice-0 | container network packet retransmission

🕒 00:30 - 01:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • frontend-2 | container read I/O load

🕒 01:30 - 02:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (2 fault(s)):
    • paymentservice-0 | container network packet retransmission
    • adservice-0 | container process termination

🕒 02:00 - 02:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • node-2 | node disk space consumption

🕒 02:30 - 03:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • cartservice | container network packet corruption

🕒 03:00 - 03:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • node-5 | node disk space consumption

🕒 03:30 - 04:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • recommendationservice-1 | container process termination

🕒 04:00 - 04:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • emailservice | container read I/O load

🕒 04:30 - 05:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • cartservice | container packet loss

🕒 05:00 - 05:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • productcatalogservice-0 | container write I/O load

🕒 06:00 - 06:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • node-4 | node memory consumption

🕒 06:30 - 07:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • recommendationservice-1 | container network packet corruption

🕒 07:00 - 07:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • frontend | container network latency

🕒 07:30 - 08:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • node-4 | node CPU load

🕒 08:00 - 08:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • productcatalogservice | container packet loss

🕒 08:30 - 09:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • recommendationservice2-0 | container write I/O load
    • node-6 | node CPU load
📅 2022-03-21 (1 fault(s)):
    • node-6 | node disk write I/O consumption

🕒 09:00 - 09:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • cartservice2-0 | container memory load
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 09:30 - 10:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • productcatalogservice | container network packet corruption
📅 2022-03-21 (1 fault(s)):
    • shippingservice-1 | container memory load

🕒 10:00 - 10:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (2 fault(s)):
    • node-6 | node CPU load
    • frontend | container memory load

🕒 10:30 - 11:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • currencyservice-0 | container read I/O load
    • node-4 | node disk read I/O consumption
📅 2022-03-21 (1 fault(s)):
    • adservice | container read I/O load

🕒 11:00 - 11:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • recommendationservice2-0 | container CPU load
    • frontend-0 | container CPU load
📅 2022-03-21 (2 fault(s)):
    • checkoutservice | container packet loss
    • node-6 | node disk read I/O consumption

🕒 11:30 - 12:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • adservice-1 | container network packet retransmission
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 12:00 - 12:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • currencyservice | container read I/O load
📅 2022-03-21 (1 fault(s)):
    • node-5 | node disk write I/O consumption

🕒 12:30 - 13:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • shippingservice | container read I/O load
    • node-4 | node CPU load
📅 2022-03-21 (1 fault(s)):
    • shippingservice-0 | container network packet corruption

🕒 13:00 - 13:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-5 | node disk read I/O consumption
📅 2022-03-21 (1 fault(s)):
    • shippingservice-2 | container read I/O load

🕒 13:30 - 14:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (2 fault(s)):
    • recommendationservice-0 | container network packet retransmission
    • frontend-2 | container packet loss

🕒 14:00 - 14:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • checkoutservice | container read I/O load
    • recommendationservice | container network latency
📅 2022-03-21 (1 fault(s)):
    • productcatalogservice-1 | container packet loss

🕒 14:30 - 15:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • emailservice-1 | container write I/O load
📅 2022-03-21 (2 fault(s)):
    • productcatalogservice2-0 | container memory load
    • frontend | container CPU load

🕒 15:00 - 15:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • checkoutservice-0 | container CPU load
    • emailservice | container network packet retransmission
📅 2022-03-21 (1 fault(s)):
    • paymentservice | container process termination

🕒 15:30 - 16:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (2 fault(s)):
    • node-2 | node CPU load
    • node-6 | node CPU spike
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 16:00 - 16:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • emailservice-0 | container network packet retransmission
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 16:30 - 17:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • cartservice-0 | container network latency
📅 2022-03-21 (1 fault(s)):
    • node-1 | node disk space consumption

🕒 17:30 - 18:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • adservice2-0 | container network latency
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 18:00 - 18:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (2 fault(s)):
    • checkoutservice | container network packet corruption
    • adservice | container packet loss

🕒 18:30 - 19:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-2 | node disk space consumption
📅 2022-03-21 (1 fault(s)):
    • node-6 | node memory consumption

🕒 19:00 - 19:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-3 | node CPU spike
📅 2022-03-21 (1 fault(s)):
    • shippingservice | container memory load

🕒 19:30 - 20:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • recommendationservice-2 | container memory load
📅 2022-03-21 (1 fault(s)):
    • paymentservice | container CPU load

🕒 20:00 - 20:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • checkoutservice-2 | container network packet corruption

🕒 20:30 - 21:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • productcatalogservice | container network packet retransmission
📅 2022-03-21 (1 fault(s)):
    • node-2 | node memory consumption

🕒 21:00 - 21:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-6 | node disk write I/O consumption
📅 2022-03-21 (1 fault(s)):
    • node-6 | node disk space consumption

🕒 21:30 - 22:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-6 | node disk write I/O consumption
📅 2022-03-21 (2 fault(s)):
    • frontend-2 | container read I/O load
    • node-5 | node memory consumption

🕒 22:00 - 22:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • recommendationservice | container write I/O load
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 22:30 - 23:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • shippingservice | container network packet retransmission
📅 2022-03-21 (1 fault(s)):
    • frontend | container network packet corruption

🕒 23:00 - 23:30
------------------------------------------------------------------------------------------
📅 2022-03-20 (1 fault(s)):
    • node-6 | node disk write I/O consumption
📅 2022-03-21 (0 fault(s)):
    (无故障)

🕒 23:30 - 00:00
------------------------------------------------------------------------------------------
📅 2022-03-20 (0 fault(s)):
    (无故障)
📅 2022-03-21 (1 fault(s)):
    • checkoutservice2-0 | container CPU load

✅ 对齐分析完成。
```