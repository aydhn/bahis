from src.core.exception_guardian import ExceptionTaxonomy

def test_cache():
    # 1. Clear cache
    ExceptionTaxonomy._CACHE.clear()

    # 2. Classify a known type
    try:
        1 / 0
    except ZeroDivisionError as e:
        cat = ExceptionTaxonomy.classify(e)
        assert cat == "math"
        assert ZeroDivisionError in ExceptionTaxonomy._CACHE
        assert ExceptionTaxonomy._CACHE[ZeroDivisionError] == "math"

    print("Test passed: Caching works perfectly!")

if __name__ == "__main__":
    test_cache()
