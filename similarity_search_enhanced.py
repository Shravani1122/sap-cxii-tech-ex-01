"""
Enhanced Product Similarity Search with Dimensionality Reduction
This is an advanced version that includes PCA for dimensionality reduction
and multimodal similarity (text + image features).
"""

import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import logging
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
import pickle
from io import BytesIO

# Setup logging first (before any usage)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image processing imports
try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm/torch not available - image features disabled")

# All colors used for one-hot encoding — fixed vocabulary so the index
# dimension is stable regardless of which colors appear in a given dataset.
COLOR_VOCABULARY = [
    'black', 'white', 'red', 'blue', 'green', 'yellow', 'pink',
    'purple', 'orange', 'brown', 'grey', 'gray', 'beige', 'navy',
    'maroon', 'gold', 'silver', 'cream', 'tan', 'olive', 'teal',
    'burgundy', 'khaki', 'coral', 'mint', 'lavender', 'peach', 'unknown',
]


class ImageFeatureExtractor:
    """
    Async image feature extractor using EfficientNet-B0 via timm.
    
    Features:
    - Async image downloading with timeout and retry
    - CNN-based feature extraction (EfficientNet-B0)
    - Disk-based caching of image vectors
    - Graceful fallback when images are unavailable
    """
    
    def __init__(
        self,
        cache_dir: str = "image_cache",
        model_name: str = "efficientnet_b0",
        timeout: int = 10,
        max_retries: int = 2,
    ):
        """
        Initialize image feature extractor.
        
        Args:
            cache_dir: Directory to cache downloaded images and vectors
            model_name: timm model name (default: efficientnet_b0)
            timeout: HTTP request timeout in seconds
            max_retries: Maximum number of download retries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.vector_cache_path = self.cache_dir / "image_vectors.pkl"
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Load or initialize vector cache
        if self.vector_cache_path.exists():
            with open(self.vector_cache_path, 'rb') as f:
                self.vector_cache = pickle.load(f)
            logger.info(f"Loaded {len(self.vector_cache)} cached image vectors")
        else:
            self.vector_cache = {}
        
        # Initialize model if available
        self.model = None
        self.transform = None
        self.feature_dim = 1280  # EfficientNet-B0 output dimension
        
        if TIMM_AVAILABLE:
            try:
                self.model = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=0,  # Remove classification head
                )
                self.model.eval()
                
                # Standard ImageNet preprocessing
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
                logger.info(f"Initialized {model_name} for image features")
            except Exception as e:
                logger.error(f"Failed to initialize image model: {e}")
                self.model = None
    
    async def download_image(
        self,
        url: str,
        session: aiohttp.ClientSession,
    ) -> Optional[Image.Image]:
        """
        Download image from URL with timeout and retry.
        
        Args:
            url: Image URL
            session: aiohttp session
            
        Returns:
            PIL Image or None if download fails
        """
        for attempt in range(self.max_retries):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        image = Image.open(BytesIO(image_data)).convert('RGB')
                        return image
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.debug(f"Failed to download {url}: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
        return None
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract feature vector from image using CNN.
        
        Args:
            image: PIL Image
            
        Returns:
            Feature vector of shape (feature_dim,)
        """
        if self.model is None or self.transform is None:
            return np.zeros(self.feature_dim, dtype='float32')
        
        try:
            with torch.no_grad():
                img_tensor = self.transform(image).unsqueeze(0)
                features = self.model(img_tensor)
                return features.squeeze().numpy().astype('float32')
        except Exception as e:
            logger.debug(f"Feature extraction failed: {e}")
            return np.zeros(self.feature_dim, dtype='float32')
    
    async def get_image_vector(
        self,
        product_id: str,
        image_url: str,
        session: aiohttp.ClientSession,
    ) -> np.ndarray:
        """
        Get image feature vector with caching.
        
        Args:
            product_id: Product unique ID
            image_url: Image URL
            session: aiohttp session
            
        Returns:
            Feature vector (zeros if image unavailable)
        """
        # Check cache first
        if product_id in self.vector_cache:
            return self.vector_cache[product_id]
        
        # Download and extract features
        image = await self.download_image(image_url, session)
        if image is None:
            vector = np.zeros(self.feature_dim, dtype='float32')
        else:
            vector = self.extract_features(image)
        
        # Cache the result
        self.vector_cache[product_id] = vector
        return vector
    
    async def batch_extract_features(
        self,
        products: List[Tuple[str, str]],
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for multiple products in parallel.
        
        Args:
            products: List of (product_id, image_url) tuples
            
        Returns:
            Dict mapping product_id to feature vector
        """
        if not TIMM_AVAILABLE or self.model is None:
            logger.warning("Image features disabled - returning zero vectors")
            return {
                pid: np.zeros(self.feature_dim, dtype='float32')
                for pid, _ in products
            }
        
        results = {}
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.get_image_vector(pid, url, session)
                for pid, url in products
            ]
            vectors = await asyncio.gather(*tasks)
            results = dict(zip([pid for pid, _ in products], vectors))
        
        # Save updated cache
        self.save_cache()
        return results
    
    def save_cache(self):
        """Persist vector cache to disk."""
        try:
            with open(self.vector_cache_path, 'wb') as f:
                pickle.dump(self.vector_cache, f)
            logger.debug(f"Saved {len(self.vector_cache)} image vectors to cache")
        except Exception as e:
            logger.error(f"Failed to save image cache: {e}")
    
    def get_feature_dim(self) -> int:
        """Return the dimensionality of image feature vectors."""
        return self.feature_dim


class EnhancedProductSimilaritySearch:
    """
    Enhanced similarity search with dimensionality reduction and multimodal support.

    Improvements over base version:
    1. PCA dimensionality reduction for faster search
    2. Color/brand/delivery one-hot encoding (proper categorical treatment)
    3. Multimodal features (text + image with CNN)
    4. Configurable feature weights
    5. Better memory efficiency
    6. get_similarity_score raises ValueError for missing products
    7. find_similar_products breaks ties by rating (desc) then sales_price (asc)
    8. Async image downloading with graceful fallback
    """

    def __init__(
        self,
        data_path: str,
        use_pca: bool = True,
        pca_components: int = 50,
        use_image_features: bool = True,
        image_weight: float = 0.15,
    ):
        """
        Initialize enhanced similarity search.

        Args:
            data_path: Path to LDJSON dataset
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components (if use_pca=True)
            use_image_features: Whether to extract and use image features
            image_weight: Weight for image features in similarity calculation
        """
        self.data_path = data_path
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.use_image_features = use_image_features and TIMM_AVAILABLE

        self.df = None
        self.index = None
        self.feature_matrix = None
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}

        # Feature weights — adjusted to include image features
        # When images are enabled, we reduce text weight slightly
        if self.use_image_features:
            self.weights = {
                'numerical': 0.20,
                'categorical': 0.15,
                'text': 0.35,
                'color': 0.15,
                'image': image_weight,
            }
        else:
            self.weights = {
                'numerical': 0.25,
                'categorical': 0.20,
                'text': 0.45,
                'color': 0.10,
                'image': 0.0,
            }

        # Scalers and vectorizers
        self.scaler = StandardScaler()
        # NOTE: All categorical features (brand, delivery, color) now use
        # one-hot encoding to avoid false ordinal relationships
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
        )
        self.pca = PCA(n_components=pca_components) if use_pca else None
        
        # Image feature extractor
        self.image_extractor = None
        if self.use_image_features:
            self.image_extractor = ImageFeatureExtractor()
            logger.info("Image features enabled")
        else:
            logger.info("Image features disabled")

        self._load_and_preprocess_data()
        self._build_faiss_index()

    # ------------------------------------------------------------------
    # Color helpers
    # ------------------------------------------------------------------

    def _extract_color_from_name(self, product_name: str) -> str:
        """Return the first matching color keyword found in product_name."""
        if pd.isna(product_name):
            return 'unknown'
        name_lower = str(product_name).lower()
        for color in COLOR_VOCABULARY:
            if color == 'unknown':
                continue
            if color in name_lower:
                return color
        return 'unknown'

    def _encode_color_onehot(self, color_series: pd.Series) -> np.ndarray:
        """
        One-hot encode a Series of color strings against COLOR_VOCABULARY.

        Using a fixed vocabulary (rather than pd.get_dummies) guarantees that
        the output dimension is constant regardless of which colors happen to
        appear in the current dataset, which matters for index stability.

        Returns shape (n_products, len(COLOR_VOCABULARY)) float32 array.
        """
        color_to_idx = {c: i for i, c in enumerate(COLOR_VOCABULARY)}
        n = len(color_series)
        k = len(COLOR_VOCABULARY)
        matrix = np.zeros((n, k), dtype='float32')
        for row_i, color in enumerate(color_series):
            col_i = color_to_idx.get(color, color_to_idx['unknown'])
            matrix[row_i, col_i] = 1.0
        return matrix

    # ------------------------------------------------------------------
    # Data loading and feature engineering
    # ------------------------------------------------------------------

    def _load_and_preprocess_data(self):
        """Load and preprocess dataset."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_json(self.data_path, lines=True)
        logger.info(f"Loaded {len(self.df)} products")

        self.product_id_to_idx = {
            pid: idx for idx, pid in enumerate(self.df['uniq_id'])
        }
        self.idx_to_product_id = {
            idx: pid for pid, idx in self.product_id_to_idx.items()
        }

        self._prepare_features()

    def _prepare_features(self):
        """Extract and engineer features."""
        logger.info("Preparing enhanced features...")

        # ---- Numerical ----
        sales_price_numeric = pd.to_numeric(
            self.df['sales_price'].astype(str).str.replace(',', ''),
            errors='coerce',
        )
        sales_price_median = sales_price_numeric.median()
        if pd.isna(sales_price_median):
            sales_price_median = 0
        self.df['sales_price_clean'] = sales_price_numeric.fillna(sales_price_median)

        weight_numeric = pd.to_numeric(self.df['weight'], errors='coerce')
        weight_numeric = weight_numeric.replace(999999999, np.nan)
        weight_median = weight_numeric.median()
        if pd.isna(weight_median):
            weight_median = 0
        self.df['weight_clean'] = weight_numeric.fillna(weight_median)

        rating_numeric = pd.to_numeric(self.df['rating'], errors='coerce')
        self.df['rating_clean'] = rating_numeric.fillna(3.0)

        # ---- Categorical ----
        self.df['brand_clean'] = self.df['brand'].fillna('unknown')
        self.df['delivery_type_clean'] = self.df['delivery_type'].fillna('unknown')

        # ---- Color (extracted from name, stored for tie-breaking display) ----
        self.df['color_extracted'] = self.df['product_name'].apply(
            self._extract_color_from_name
        )
        logger.info(
            f"Top extracted colors:\n{self.df['color_extracted'].value_counts().head()}"
        )

        # ---- Text ----
        self.df['text_features'] = (
            self.df['product_name'].fillna('') + ' '
            + self.df['meta_keywords'].fillna('')
        )

        logger.info("Enhanced feature preparation complete")

    def _encode_categorical_onehot(
        self,
        series: pd.Series,
        vocabulary: Optional[List[str]] = None,
        max_categories: int = 100,
    ) -> np.ndarray:
        """
        One-hot encode a categorical Series.
        
        Args:
            series: Pandas Series to encode
            vocabulary: Fixed vocabulary (if None, use top max_categories)
            max_categories: Maximum number of categories to keep
            
        Returns:
            One-hot encoded matrix of shape (n, vocab_size)
        """
        if vocabulary is None:
            # Use top N most frequent values
            value_counts = series.value_counts()
            vocabulary = value_counts.head(max_categories).index.tolist()
            if 'unknown' not in vocabulary:
                vocabulary.append('unknown')
        
        cat_to_idx = {cat: i for i, cat in enumerate(vocabulary)}
        n = len(series)
        k = len(vocabulary)
        matrix = np.zeros((n, k), dtype='float32')
        
        for row_i, value in enumerate(series):
            col_i = cat_to_idx.get(value, cat_to_idx.get('unknown', 0))
            matrix[row_i, col_i] = 1.0
        
        return matrix

    def _build_faiss_index(self):
        """Build FAISS index with optional PCA and image features."""
        logger.info("Building enhanced FAISS index...")

        # Numerical
        numerical_features = self.df[
            ['sales_price_clean', 'weight_clean', 'rating_clean']
        ].values
        numerical_features_scaled = self.scaler.fit_transform(numerical_features)

        # Categorical — now using one-hot encoding for all categorical features
        # This fixes the false ordinal relationship issue
        # Store vocabularies for persistence/reproducibility
        brand_value_counts = self.df['brand_clean'].value_counts()
        self.brand_vocabulary = brand_value_counts.head(50).index.tolist()
        if 'unknown' not in self.brand_vocabulary:
            self.brand_vocabulary.append('unknown')
        
        delivery_value_counts = self.df['delivery_type_clean'].value_counts()
        self.delivery_vocabulary = delivery_value_counts.head(10).index.tolist()
        if 'unknown' not in self.delivery_vocabulary:
            self.delivery_vocabulary.append('unknown')
        
        brand_features = self._encode_categorical_onehot(
            self.df['brand_clean'],
            vocabulary=self.brand_vocabulary,
        )
        delivery_features = self._encode_categorical_onehot(
            self.df['delivery_type_clean'],
            vocabulary=self.delivery_vocabulary,
        )
        categorical_features = np.hstack([brand_features, delivery_features])
        logger.info(
            f"Categorical one-hot dimensions: "
            f"brand={brand_features.shape[1]}, "
            f"delivery={delivery_features.shape[1]}"
        )

        # Color — one-hot with fixed vocabulary
        color_features = self._encode_color_onehot(self.df['color_extracted'])
        logger.info(f"Color one-hot dimension: {color_features.shape[1]}")

        # TF-IDF text
        text_features = self.tfidf_vectorizer.fit_transform(
            self.df['text_features']
        ).toarray()

        # Image features (if enabled)
        # Only allocate and extract if actually using images to save ~150MB memory
        blocks_to_combine = [
            numerical_features_scaled * self.weights['numerical'],
            categorical_features * self.weights['categorical'],
            text_features * self.weights['text'],
            color_features * self.weights['color'],
        ]
        
        if self.use_image_features and self.image_extractor is not None:
            logger.info("Extracting image features (this may take a while)...")
            image_features = self._extract_image_features()
            logger.info(f"Image feature dimension: {image_features.shape[1]}")
            blocks_to_combine.append(image_features * self.weights['image'])
        else:
            logger.info("Image features disabled - skipping allocation")

        # Combine with weights
        # NOTE: This weighting approach has limitations with PCA - weights are
        # approximate after PCA transformation. For precise control, consider
        # weighted combination of per-block similarities instead.
        combined_features = np.hstack(blocks_to_combine).astype('float32')

        logger.info(f"Combined feature dimension: {combined_features.shape[1]}")

        # Optional PCA
        if self.use_pca and combined_features.shape[1] > self.pca_components:
            logger.info(
                f"Applying PCA: {combined_features.shape[1]} → "
                f"{self.pca_components} dimensions"
            )
            self.feature_matrix = self.pca.fit_transform(
                combined_features
            ).astype('float32')
            explained_variance = self.pca.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance: {explained_variance:.2%}")
        else:
            self.feature_matrix = combined_features

        # L2-normalise so inner product == cosine similarity
        faiss.normalize_L2(self.feature_matrix)

        dimension = self.feature_matrix.shape[1]
        # IndexFlatIP performs EXACT search (not approximate)
        # It scans all vectors - fine for 30k products (<10ms queries)
        # For true ANN at scale (>1M), consider IndexIVFFlat or IndexHNSWFlat
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.feature_matrix)

        logger.info(
            f"FAISS index built (exact search): {self.index.ntotal} vectors, {dimension}D"
        )

    def _extract_image_features(self) -> np.ndarray:
        """
        Extract image features for all products.
        
        Returns:
            Image feature matrix of shape (n_products, image_dim)
        """
        if self.image_extractor is None:
            return np.zeros((len(self.df), 1280), dtype='float32')
        
        # Prepare list of (product_id, image_url) tuples
        products = []
        for idx, row in self.df.iterrows():
            product_id = row['uniq_id']
            image_url = row.get('image', '')
            if pd.notna(image_url) and str(image_url).startswith('http'):
                products.append((product_id, str(image_url)))
            else:
                products.append((product_id, ''))  # Will get zero vector
        
        # Extract features synchronously (avoid event loop conflicts)
        # Use nest_asyncio to allow nested event loops
        # Use ThreadPoolExecutor to avoid event loop conflicts
        # This is production-safe and works with FastAPI's async context
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            vector_dict = pool.submit(
                asyncio.run,
                self.image_extractor.batch_extract_features(products)
            ).result()
        
        # Build feature matrix in correct order
        feature_matrix = np.zeros(
            (len(self.df), self.image_extractor.get_feature_dim()),
            dtype='float32'
        )
        for idx, row in self.df.iterrows():
            product_id = row['uniq_id']
            feature_matrix[idx] = vector_dict.get(
                product_id,
                np.zeros(self.image_extractor.get_feature_dim(), dtype='float32')
            )
        
        # Log statistics
        non_zero_count = np.count_nonzero(feature_matrix.sum(axis=1))
        logger.info(
            f"Image features extracted: {non_zero_count}/{len(self.df)} "
            f"products have valid images ({non_zero_count/len(self.df)*100:.1f}%)"
        )
        
        return feature_matrix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_product_vector(self, product_id: str) -> Optional[np.ndarray]:
        """
        Get feature vector for a product.
        
        Note: No caching needed - NumPy array indexing is already O(1).
        Using @lru_cache on instance methods is problematic as it caches
        on (self, product_id) and self is not hashable.
        """
        if product_id not in self.product_id_to_idx:
            return None
        idx = self.product_id_to_idx[product_id]
        return self.feature_matrix[idx]

    def find_similar_products(
        self,
        product_id: str,
        num_similar: int = 10,
        exclude_self: bool = True,
    ) -> List[str]:
        """
        Find similar products using enhanced FAISS search.

        Tie-breaking: when two candidates have the same cosine similarity score
        (after rounding to 6 decimal places) they are sorted by:
          1. rating_clean descending  (higher-rated first)
          2. sales_price_clean ascending  (cheaper first among equal ratings)

        Args:
            product_id: Product unique ID
            num_similar: Number of similar products to return
            exclude_self: Exclude the query product from results

        Returns:
            List of similar product IDs sorted by descending similarity.

        Raises:
            ValueError: If product_id is not in the dataset.
        """
        if product_id not in self.product_id_to_idx:
            raise ValueError(f"Product ID '{product_id}' not found")

        query_idx = self.product_id_to_idx[product_id]
        query_vector = self.feature_matrix[query_idx : query_idx + 1]

        # Fetch extra candidates to allow for self-exclusion
        k = num_similar + 1 if exclude_self else num_similar
        distances, indices = self.index.search(query_vector, k)

        # Build candidate list: (product_id, score, rating, price)
        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            pid = self.idx_to_product_id[idx]
            if exclude_self and pid == product_id:
                continue
            row = self.df.iloc[idx]
            candidates.append(
                (pid, float(dist), float(row['rating_clean']), float(row['sales_price_clean']))
            )
            if len(candidates) >= num_similar:
                break

        # Stable tie-breaking: sort by (score desc, rating desc, price asc).
        # FAISS already returns results in score order but this makes ties
        # deterministic and satisfies the spec requirement explicitly.
        candidates.sort(key=lambda x: (-round(x[1], 6), -x[2], x[3]))

        return [pid for pid, *_ in candidates]

    def get_product_details(self, product_id: str) -> Optional[Dict]:
        """Return a dict of product attributes, or None if not found."""
        if product_id not in self.product_id_to_idx:
            return None
        idx = self.product_id_to_idx[product_id]
        return self.df.iloc[idx].to_dict()

    def get_similarity_score(self, product_id1: str, product_id2: str) -> float:
        """
        Calculate cosine similarity between two products.

        Returns:
            float in [-1, 1] (typically [0, 1] after L2 normalisation).

        Raises:
            ValueError: If either product_id is not in the dataset.
                        Previously this returned 0.0, which was ambiguous with
                        a genuine zero-similarity score.
        """
        if product_id1 not in self.product_id_to_idx:
            raise ValueError(f"Product ID '{product_id1}' not found")
        if product_id2 not in self.product_id_to_idx:
            raise ValueError(f"Product ID '{product_id2}' not found")

        idx1 = self.product_id_to_idx[product_id1]
        idx2 = self.product_id_to_idx[product_id2]

        vec1 = self.feature_matrix[idx1]
        vec2 = self.feature_matrix[idx2]

        return float(np.dot(vec1, vec2))

    def get_feature_importance(self) -> Dict[str, float]:
        """Return the configured feature weights."""
        return self.weights.copy()

    def get_pca_explained_variance(self) -> Optional[float]:
        """
        Return the cumulative explained variance ratio from PCA, or None if
        PCA was not applied.  Useful for surfacing via the /health endpoint.
        """
        if self.pca is None or not hasattr(self.pca, 'explained_variance_ratio_'):
            return None
        return float(self.pca.explained_variance_ratio_.sum())


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_enhanced_instance: Optional[EnhancedProductSimilaritySearch] = None


def get_enhanced_similarity_search(
    data_path: str = None,
    use_pca: bool = True,
    pca_components: int = 50,
) -> EnhancedProductSimilaritySearch:
    """Get or create the module-level singleton instance."""
    global _enhanced_instance

    if _enhanced_instance is None:
        if data_path is None:
            data_path = (
                'data/marketing_sample_for_amazon_com-'
                'amazon_fashion_products__20200201_20200430__30k_data.ldjson'
            )
        _enhanced_instance = EnhancedProductSimilaritySearch(
            data_path,
            use_pca=use_pca,
            pca_components=pca_components,
            use_image_features=False,  # Disabled by default to avoid event loop issues
        )

    return _enhanced_instance


if __name__ == "__main__":
    print("Testing Enhanced Similarity Search...")
    print("=" * 60)

    search = EnhancedProductSimilaritySearch(
        'data/marketing_sample_for_amazon_com-'
        'amazon_fashion_products__20200201_20200430__30k_data.ldjson',
        use_pca=True,
        pca_components=50,
    )

    test_id = search.df['uniq_id'].iloc[0]
    print(f"\nQuery product : {test_id}")
    print(f"Name          : {search.df.iloc[0]['product_name']}")
    print(f"Color         : {search.df.iloc[0]['color_extracted']}")

    similar = search.find_similar_products(test_id, num_similar=5)
    print(f"\nTop 5 similar products:")
    for i, pid in enumerate(similar, 1):
        details = search.get_product_details(pid)
        score = search.get_similarity_score(test_id, pid)
        print(f"\n{i}. {pid}")
        print(f"   Name      : {str(details['product_name'])[:60]}...")
        print(f"   Brand     : {details['brand']}")
        print(f"   Color     : {details['color_extracted']}")
        print(f"   Price     : {details['sales_price']}")
        print(f"   Rating    : {details['rating_clean']}")
        print(f"   Similarity: {score:.4f}")

    print(f"\nFeature importance:")
    for feature, weight in search.get_feature_importance().items():
        print(f"  {feature}: {weight:.2%}")

    pca_var = search.get_pca_explained_variance()
    if pca_var is not None:
        print(f"\nPCA explained variance: {pca_var:.2%}")

    print("\n" + "=" * 60)
    print("Test complete!")
