"""Presentation-only controls for the same-origin Community Cloud wrapper."""
import streamlit as st


def hide_creator_profile() -> None:
    # Community Cloud renders the creator link outside the app iframe. A
    # same-origin stylesheet hides only that link, including when added later.
    # Keep the hosting badge and sidebar controls intact. This is a UI preference,
    # not access control: the creator's public profile still exists.
    st.html(
        """<script>
        (() => {
            try {
                const host = window.parent;
                if (!host.location.hostname.endsWith('.streamlit.app')) return;
                const doc = host.document;
                const id = 'bladescope-hide-creator-profile';
                if (doc.getElementById(id)) return;
                const style = doc.createElement('style');
                style.id = id;
                style.textContent = 'a:has(img[data-testid="appCreatorAvatar"]) { display: none !important; }';
                doc.head.appendChild(style);
            } catch (_) {
                // An unrelated cross-origin embed cannot change its host page.
            }
        })();
        </script>""",
        unsafe_allow_javascript=True,
    )
