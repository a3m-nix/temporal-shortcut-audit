import os
import importlib.util, sys
from pathlib import Path
B=Path(os.environ["TEMPORAL_AUDIT_ROOT"] + "")
spec=importlib.util.spec_from_file_location("p24", B/"0-temporal-feature-audit-v4-allminmax-3seed10fold/24-thresholdfree-multibudget-translation-posthoc-v1.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.AUDIT_ROOT=B/"temporal_audit_temporal_upgrade_e2e_v1_2_edgefix"
sys.argv=["24","--output-root",str(B/"temporal_audit_posthoc_strengthening_v1/06_thresholdfree_multibudget_translation_edgefix")]
m.main()
