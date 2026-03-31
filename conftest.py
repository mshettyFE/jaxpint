import os
import hypothesis

hypothesis.settings.register_profile("interactive", deadline=None)
hypothesis.settings.register_profile(
    "ci", deadline=None, print_blob=True, derandomize=True
)
hypothesis.settings.register_profile(
    "fuzzing", deadline=None, print_blob=True, max_examples=1000
)
default = (
    "fuzzing"
    if os.environ.get("HYPOTHESIS_PROFILE") == "fuzzing"
    else "interactive"
)
hypothesis.settings.load_profile(default)
