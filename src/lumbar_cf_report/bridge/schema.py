LEVELS=['L1/2','L2/3','L3/4','L4/5','L5/S1']
TASKS=['disc','stenosis','nerve']
MAIN_SLOT_NAMES=['lordosis']+[f'{t}:{lv}' for t in TASKS for lv in LEVELS]
STRUCTURE_TERMS=['facet_degeneration','ligamentum_flavum_hypertrophy','endplate_change','spondylolisthesis','scoliosis','vertebral_fracture','marrow_abnormality','other_structure_abnormality']

def disease_slot_index(level_idx,task_idx):return 1+task_idx*5+level_idx
