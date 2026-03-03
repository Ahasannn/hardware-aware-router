// Pre-processed data from evaluation_dataset_processed_full_with_umr_irt.csv
//
// Architecture: HW-Router = any quality predictor + hardware cost predictor (MLP).
// The hardware cost predictor is a *plugin* — it replaces static pricing with
// real-time latency predictions and can be composed with any quality router.
//
// Each lambda row has 6 router variants:
//   carrot     = CARROT quality + static cost (price/token)
//   irt        = IRT quality   + static cost
//   umr        = UMR quality   + static cost
//   carrot_hw  = CARROT quality + hardware cost MLP  ← CARROT made hardware-aware
//   irt_hw     = IRT quality   + hardware cost MLP  ← IRT made hardware-aware (paper default)
//   umr_hw     = UMR quality   + hardware cost MLP  ← UMR made hardware-aware
//
// Score formula: S = λ·Q(x) − (1−λ)·C(x,h)
//   Quality-only:       C = static price/token
//   Hardware-aware (+HW): C = MLP(hardware_state) → predicts TTFT + TPOT

const LAMBDA_SWEEP = [
  {"lambda":0.0,"carrot":{"quality":0.6201,"latency":40.0767,"slo_e2e":0.5246,"slo_ttft":0.5258},"irt":{"quality":0.5251,"latency":24.2204,"slo_e2e":0.9674,"slo_ttft":0.9852},"umr":{"quality":0.5251,"latency":24.2204,"slo_e2e":0.9674,"slo_ttft":0.9852},"carrot_hw":{"quality":0.5809,"latency":11.2378,"slo_e2e":0.9994,"slo_ttft":0.9994},"irt_hw":{"quality":0.5809,"latency":11.2378,"slo_e2e":0.9994,"slo_ttft":0.9994},"umr_hw":{"quality":0.5809,"latency":11.2378,"slo_e2e":0.9994,"slo_ttft":0.9994}},
  {"lambda":0.1,"carrot":{"quality":0.6272,"latency":40.3854,"slo_e2e":0.5169,"slo_ttft":0.5205},"irt":{"quality":0.5251,"latency":24.2204,"slo_e2e":0.9674,"slo_ttft":0.9852},"umr":{"quality":0.5251,"latency":24.2204,"slo_e2e":0.9674,"slo_ttft":0.9852},"carrot_hw":{"quality":0.5816,"latency":11.2751,"slo_e2e":0.9994,"slo_ttft":0.9988},"irt_hw":{"quality":0.5815,"latency":11.2263,"slo_e2e":0.9994,"slo_ttft":0.9994},"umr_hw":{"quality":0.5830,"latency":11.2745,"slo_e2e":0.9994,"slo_ttft":0.9994}},
  {"lambda":0.2,"carrot":{"quality":0.6352,"latency":41.4711,"slo_e2e":0.4926,"slo_ttft":0.4997},"irt":{"quality":0.5251,"latency":24.1969,"slo_e2e":0.9674,"slo_ttft":0.9852},"umr":{"quality":0.5251,"latency":24.2204,"slo_e2e":0.9674,"slo_ttft":0.9852},"carrot_hw":{"quality":0.5829,"latency":11.3783,"slo_e2e":0.9988,"slo_ttft":0.9982},"irt_hw":{"quality":0.5835,"latency":11.2473,"slo_e2e":0.9994,"slo_ttft":0.9994},"umr_hw":{"quality":0.5831,"latency":11.3260,"slo_e2e":0.9988,"slo_ttft":0.9988}},
  {"lambda":0.3,"carrot":{"quality":0.6432,"latency":42.1275,"slo_e2e":0.4807,"slo_ttft":0.4890},"irt":{"quality":0.5412,"latency":23.5964,"slo_e2e":0.9680,"slo_ttft":0.9852},"umr":{"quality":0.5598,"latency":21.7283,"slo_e2e":0.9727,"slo_ttft":0.9875},"carrot_hw":{"quality":0.5839,"latency":11.5604,"slo_e2e":0.9988,"slo_ttft":0.9982},"irt_hw":{"quality":0.5831,"latency":11.2556,"slo_e2e":0.9988,"slo_ttft":0.9988},"umr_hw":{"quality":0.5845,"latency":11.5093,"slo_e2e":0.9964,"slo_ttft":0.9964}},
  {"lambda":0.4,"carrot":{"quality":0.6530,"latency":42.8587,"slo_e2e":0.4659,"slo_ttft":0.4807},"irt":{"quality":0.5631,"latency":22.0798,"slo_e2e":0.9727,"slo_ttft":0.9875},"umr":{"quality":0.5661,"latency":20.6768,"slo_e2e":0.9751,"slo_ttft":0.9887},"carrot_hw":{"quality":0.5909,"latency":12.1384,"slo_e2e":0.9917,"slo_ttft":0.9935},"irt_hw":{"quality":0.5848,"latency":11.4334,"slo_e2e":0.9976,"slo_ttft":0.9982},"umr_hw":{"quality":0.5954,"latency":13.8111,"slo_e2e":0.9620,"slo_ttft":0.9638}},
  {"lambda":0.5,"carrot":{"quality":0.6573,"latency":43.8804,"slo_e2e":0.4469,"slo_ttft":0.4647},"irt":{"quality":0.5768,"latency":20.0674,"slo_e2e":0.9769,"slo_ttft":0.9899},"umr":{"quality":0.5676,"latency":20.2221,"slo_e2e":0.9769,"slo_ttft":0.9899},"carrot_hw":{"quality":0.6061,"latency":14.4255,"slo_e2e":0.9614,"slo_ttft":0.9715},"irt_hw":{"quality":0.6059,"latency":12.8591,"slo_e2e":0.9792,"slo_ttft":0.9881},"umr_hw":{"quality":0.6156,"latency":16.6946,"slo_e2e":0.9151,"slo_ttft":0.9258}},
  {"lambda":0.6,"carrot":{"quality":0.6692,"latency":44.6808,"slo_e2e":0.4344,"slo_ttft":0.4617},"irt":{"quality":0.5890,"latency":18.0587,"slo_e2e":0.9828,"slo_ttft":0.9923},"umr":{"quality":0.5677,"latency":19.8138,"slo_e2e":0.9780,"slo_ttft":0.9905},"carrot_hw":{"quality":0.6333,"latency":18.8236,"slo_e2e":0.8861,"slo_ttft":0.9104},"irt_hw":{"quality":0.6317,"latency":19.0713,"slo_e2e":0.8682,"slo_ttft":0.8950},"umr_hw":{"quality":0.6416,"latency":21.5459,"slo_e2e":0.8338,"slo_ttft":0.8605}},
  {"lambda":0.7,"carrot":{"quality":0.6732,"latency":44.5978,"slo_e2e":0.4374,"slo_ttft":0.4694},"irt":{"quality":0.5951,"latency":16.8395,"slo_e2e":0.9864,"slo_ttft":0.9929},"umr":{"quality":0.5867,"latency":18.2160,"slo_e2e":0.9448,"slo_ttft":0.9513},"carrot_hw":{"quality":0.6523,"latency":26.3682,"slo_e2e":0.7555,"slo_ttft":0.7953},"irt_hw":{"quality":0.6483,"latency":22.9543,"slo_e2e":0.8042,"slo_ttft":0.8392},"umr_hw":{"quality":0.6656,"latency":28.8010,"slo_e2e":0.7068,"slo_ttft":0.7478}},
  {"lambda":0.8,"carrot":{"quality":0.6776,"latency":44.3334,"slo_e2e":0.4427,"slo_ttft":0.4837},"irt":{"quality":0.6132,"latency":17.9433,"slo_e2e":0.9412,"slo_ttft":0.9561},"umr":{"quality":0.6398,"latency":25.9298,"slo_e2e":0.7935,"slo_ttft":0.8297},"carrot_hw":{"quality":0.6742,"latency":33.8707,"slo_e2e":0.6202,"slo_ttft":0.6718},"irt_hw":{"quality":0.6766,"latency":30.5665,"slo_e2e":0.6861,"slo_ttft":0.7323},"umr_hw":{"quality":0.6851,"latency":36.7952,"slo_e2e":0.5680,"slo_ttft":0.6208}},
  {"lambda":0.9,"carrot":{"quality":0.6837,"latency":43.8351,"slo_e2e":0.4433,"slo_ttft":0.5003},"irt":{"quality":0.6716,"latency":33.4757,"slo_e2e":0.6297,"slo_ttft":0.6772},"umr":{"quality":0.6788,"latency":41.9586,"slo_e2e":0.4777,"slo_ttft":0.5270},"carrot_hw":{"quality":0.6813,"latency":39.3363,"slo_e2e":0.5252,"slo_ttft":0.5858},"irt_hw":{"quality":0.6963,"latency":42.9023,"slo_e2e":0.4611,"slo_ttft":0.5347},"umr_hw":{"quality":0.6932,"latency":43.9646,"slo_e2e":0.4451,"slo_ttft":0.5068}},
  {"lambda":1.0,"carrot":{"quality":0.6866,"latency":43.0099,"slo_e2e":0.4593,"slo_ttft":0.5288},"irt":{"quality":0.6945,"latency":45.4531,"slo_e2e":0.4030,"slo_ttft":0.4908},"umr":{"quality":0.6950,"latency":45.7361,"slo_e2e":0.4071,"slo_ttft":0.4866},"carrot_hw":{"quality":0.6866,"latency":43.0099,"slo_e2e":0.4593,"slo_ttft":0.5288},"irt_hw":{"quality":0.6945,"latency":45.4531,"slo_e2e":0.4030,"slo_ttft":0.4908},"umr_hw":{"quality":0.6950,"latency":45.7361,"slo_e2e":0.4071,"slo_ttft":0.4866}}
];

// Key results from paper (at default lambda = 0.5, IRT used as quality component for HW-Router)
const KEY_RESULTS = [
  { router: "CARROT",             quality: 0.657, latency: 43.9, slo: 0.447, isOurs: false },
  { router: "IRT",                quality: 0.669, latency: 45.3, slo: 0.422, isOurs: false },
  { router: "UMR",                quality: 0.665, latency: 48.4, slo: 0.373, isOurs: false },
  { router: "HW-Router (IRT+HW)", quality: 0.606, latency: 12.9, slo: 0.979, isOurs: true  },
];

// Router label lookup (used by JS for card/legend labels)
const ROUTER_META = {
  carrot:    { label: "CARROT" },
  irt:       { label: "IRT"    },
  umr:       { label: "UMR"   },
  carrot_hw: { label: "CARROT + HW Cost" },
  irt_hw:    { label: "IRT + HW Cost"    },
  umr_hw:    { label: "UMR + HW Cost"    },
};
