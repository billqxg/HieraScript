import torch
import torch.utils.data as data
from torchvision import transforms
from os.path import join
from PIL import Image
import os
import networkx as nx
import numpy as np

class MultilangDataset(data.Dataset):
    def __init__(self, image_dir, input_transform=None, re_level='language'):
        super(MultilangDataset, self).__init__()
        
        self.re_level = re_level
        
        self.macro_types = []   # Level 1
        self.scripts = []       # Level 2
        self.languages = []     # Level 3
        
        self._collect_hierarchy_names(image_dir)
        self.macro_to_id = {m: i for i, m in enumerate(sorted(self.macro_types))}
        self.macro_to_id_inv = {i: m for m, i in self.macro_to_id.items()}
        self.script_to_id = {s: i + len(self.macro_types) 
                            for i, s in enumerate(sorted(self.scripts))}
        self.script_to_id_inv = {i: s for s, i in self.script_to_id.items()}
        self.lang_to_id = {l: i + len(self.macro_types) + len(self.scripts) 
                          for i, l in enumerate(sorted(self.languages))}
        self.lang_to_id_inv = {i: l for l, i in self.lang_to_id.items()}
        self.hierarchy = self._build_hierarchy_with_ids(image_dir)
        self.trees = self._build_trees()
        self.image_filenames, self.labels = self._collect_data(image_dir)
        self.input_transform = input_transform
        self.g, self.g_t, self.adj_matrix, self.to_eval, self.nodes_idx = self._compute_adj_matrix()
        
    def _collect_hierarchy_names(self, image_dir):
        for macro_type in sorted(os.listdir(image_dir)):
            macro_path = join(image_dir, macro_type)
            if not os.path.isdir(macro_path):
                continue
                
            if macro_type not in self.macro_types:
                self.macro_types.append(macro_type)
            
            for script in sorted(os.listdir(macro_path)):
                script_path = join(macro_path, script)
                if not os.path.isdir(script_path):
                    continue
                    
                if script not in self.scripts:
                    self.scripts.append(script)
                
                for language in sorted(os.listdir(script_path)):
                    lang_path = join(script_path, language)
                    if not os.path.isdir(lang_path):
                        continue
                        
                    if language not in self.languages:
                        self.languages.append(language)

    def _build_hierarchy_with_ids(self, image_dir):
        hierarchy = {}
        
        for macro_type in sorted(os.listdir(image_dir)):
            macro_path = join(image_dir, macro_type)
            if not os.path.isdir(macro_path):
                continue
            
            for script in sorted(os.listdir(macro_path)):
                script_path = join(macro_path, script)
                if not os.path.isdir(script_path):
                    continue
                
                for language in sorted(os.listdir(script_path)):
                    lang_path = join(script_path, language)
                    if not os.path.isdir(lang_path):
                        continue
                    
                    key = f"{macro_type}-{script}-{language}"
                    hierarchy[key] = {
                        'macro_type': macro_type,      # Level 1
                        'script': script,              # Level 2
                        'language': language,          # Level 3
                        'path': lang_path,
                        'macro_id': self.macro_to_id[macro_type],
                        'script_id': self.script_to_id[script],
                        'lang_id': self.lang_to_id[language]
                    }
        
        return hierarchy
    
    def _build_trees(self):
        trees = []
        for lang in sorted(self.languages):
            for key, info in self.hierarchy.items():
                if info['language'] == lang:
                    leaf_id = self.lang_to_id[lang]
                    script_idx = self.script_to_id[info['script']]
                    macro_idx = self.macro_to_id[info['macro_type']]
                    trees.append([leaf_id, script_idx, macro_idx])
                    break
        return trees
    
    def _collect_data(self, image_dir):
        images = []
        labels = []
        
        for key, info in self.hierarchy.items():
            lang_path = info['path']
            lang_id = self.lang_to_id[info['language']]
            
            for img_name in sorted(os.listdir(lang_path)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif')):
                    img_path = join(lang_path, img_name)
                    images.append(img_path)
                    labels.append(lang_id)
        
        return images, labels
    
    def _compute_adj_matrix(self):
        g = nx.DiGraph()
        for lang in sorted(self.languages):
            for key, info in self.hierarchy.items():
                if info['language'] == lang:
                    macro_id = self.macro_to_id[info['macro_type']]
                    script_id = self.script_to_id[info['script']]
                    lang_id = self.lang_to_id[lang]
                    
                    g.add_edge(lang_id, script_id)
                    g.add_edge(script_id, macro_id)
                    g.add_edge(macro_id, -1)
                    break
        
        nodes = sorted(g.nodes())
        nodes_idx = dict(zip(nodes, range(len(nodes))))
        g_t = g.reverse()
        am = nx.to_numpy_array(g, nodelist=nodes)
        to_eval = [t not in [-1] for t in nodes]
        
        return g, g_t, np.array(am), to_eval, nodes_idx
    
    def get_ancestors(self, node_id):
        ancestors = nx.ancestors(self.g_t, node_id)
        return set(ancestors) | {node_id}
    
    def get_level_from_id(self, node_id):
        if node_id < len(self.macro_types):
            return 1  # Macro Level
        elif node_id < len(self.macro_types) + len(self.scripts):
            return 2  # Script Level
        else:
            return 3  # Language Level
    
    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        target = self.labels[index]
        
        img = Image.open(imagename).convert('RGB')
        if self.input_transform:
            img = self.input_transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.image_filenames)
    
    def get_hierarchy_info(self):
        return {
            'num_macros': len(self.macro_types),
            'num_scripts': len(self.scripts),
            'num_languages': len(self.languages),
            'total_nodes': len(self.macro_types) + len(self.scripts) + len(self.languages),
            'trees': self.trees,
            'levels': 3,
            'macro_to_id': self.macro_to_id,
            'script_to_id': self.script_to_id,
            'lang_to_id': self.lang_to_id,
            'macro_to_id_inv': self.macro_to_id_inv,
            'script_to_id_inv': self.script_to_id_inv,
            'lang_to_id_inv': self.lang_to_id_inv,
            'hierarchy': self.hierarchy,
            'dataset': self
        }