/* predict_form.js
 * Drives the comprehensive manual prediction form:
 *  - Real-time AJAX analysis of username/bio/photo/posts
 *  - Derived feature display (ratios, account age, posts/day)
 *  - Data coverage tracker (sticky indicator at top)
 *  - Quick-fill demo data
 *  - Live Lookup auto-fill (legacy + extended fields)
 *  - Hidden JSON fields for pre-computed feature dicts
 */
(function () {
  'use strict';

  const form = document.getElementById('predictForm');
  if (!form) return;
  const platform = form.dataset.platform;
  const csrfToken = form.querySelector('[name=csrf_token]').value;

  // ---------- Debounce helper ----------
  function debounce(fn, ms) {
    let t = null;
    return function (...args) {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  // ---------- Inline result rendering ----------
  function renderInlineSignals(containerId, signals) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (!signals.length) { el.style.display = 'none'; el.innerHTML = ''; return; }
    el.innerHTML = signals.map(s => {
      const klass = s.severity || 'neutral';
      const icon = s.icon || (klass === 'ok' ? '✓' : klass === 'bad' ? '✗' : klass === 'warn' ? '⚠' : '·');
      return `<span class="signal ${klass}">${icon} ${s.label}</span>`;
    }).join('');
    el.style.display = '';
  }

  function badgeFromScore(label, score, lowGood) {
    // lowGood=true: smaller is better
    let sev = 'neutral';
    if (score == null) return { label: `${label}: —`, severity: 'neutral' };
    if (lowGood) {
      sev = score < 0.15 ? 'ok' : score < 0.4 ? 'warn' : 'bad';
    } else {
      sev = score > 0.7 ? 'ok' : score > 0.4 ? 'warn' : 'bad';
    }
    return { label: `${label}: ${typeof score === 'number' ? score.toFixed(2) : score}`, severity: sev };
  }

  // ---------- Username analysis ----------
  function analyzeUsername(value) {
    if (!value || value.length < 2) return;
    fetch(`/api/v1/analyze/username/${encodeURIComponent(value)}`)
      .then(r => r.json())
      .then(data => {
        if (data.error) return;
        document.getElementById('username-features-json').value = JSON.stringify(data);
        const sigs = [];
        if (data.uname_bot_pattern_count != null) {
          sigs.push({
            label: `Bot patterns: ${data.uname_bot_pattern_count}`,
            severity: data.uname_bot_pattern_count > 0 ? 'bad' : 'ok'
          });
        }
        if (data.uname_entropy != null) sigs.push({ label: `Entropy: ${data.uname_entropy.toFixed(2)}`, severity: 'neutral' });
        if (data.uname_homoglyph_count != null && data.uname_homoglyph_count > 0)
          sigs.push({ label: `Homoglyphs: ${data.uname_homoglyph_count}`, severity: 'bad' });
        if (data.uname_dict_coverage != null)
          sigs.push({ label: `Dict coverage: ${(data.uname_dict_coverage*100).toFixed(0)}%`, severity: data.uname_dict_coverage > 0.4 ? 'ok' : 'neutral' });
        if (data.uname_brand_distance != null && data.uname_brand_distance < 3 && data.uname_brand_closest)
          sigs.push({ label: `Close to brand "${data.uname_brand_closest}"`, severity: 'warn' });
        if (data.uname_pronounceable_score != null)
          sigs.push({ label: `Pronounceable: ${(data.uname_pronounceable_score*100).toFixed(0)}%`, severity: 'neutral' });
        renderInlineSignals('username-analysis-result', sigs);
      })
      .catch(() => {});
  }

  // ---------- Bio analysis ----------
  function analyzeBio(value) {
    if (!value || value.trim().length < 3) {
      document.getElementById('bio-features-json').value = '';
      const el = document.getElementById('bio-analysis-result');
      if (el) { el.style.display = 'none'; el.innerHTML = ''; }
      return;
    }
    fetch('/api/v1/analyze/bio', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
      body: JSON.stringify({ bio: value, platform })
    })
      .then(r => r.json())
      .then(data => {
        if (data.error) return;
        document.getElementById('bio-features-json').value = JSON.stringify(data);
        const sigs = [];
        if (data.bio_language) sigs.push({ label: `Lang: ${data.bio_language}`, severity: 'neutral' });
        if (data.bio_spam_score != null) sigs.push(badgeFromScore('Spam', data.bio_spam_score, true));
        if (data.bio_template_similarity != null) sigs.push(badgeFromScore('Template match', data.bio_template_similarity, true));
        if (data.bio_url_count != null && data.bio_url_count > 0)
          sigs.push({ label: `URLs: ${data.bio_url_count}`, severity: data.bio_url_count > 2 ? 'warn' : 'neutral' });
        if (data.bio_invisible_chars != null && data.bio_invisible_chars > 0)
          sigs.push({ label: `Invisible chars: ${data.bio_invisible_chars}`, severity: 'bad' });
        if (data.bio_caps_ratio != null && data.bio_caps_ratio > 0.5)
          sigs.push({ label: `Excessive caps`, severity: 'warn' });
        if (data.bio_emoji_count != null) sigs.push({ label: `Emojis: ${data.bio_emoji_count}`, severity: 'neutral' });
        if (data.bio_phone_present) sigs.push({ label: 'Phone detected', severity: 'warn' });
        renderInlineSignals('bio-analysis-result', sigs);
      })
      .catch(() => {});
  }

  // ---------- Photo analysis ----------
  function analyzeAvatarFile(file) {
    if (!file) return;
    if (file.size > 5 * 1024 * 1024) {
      const el = document.getElementById('photo-analysis-result');
      if (el) {
        el.style.display = '';
        el.innerHTML = '<span class="signal bad">✗ Image exceeds 5 MB limit</span>';
      }
      return;
    }
    const fd = new FormData();
    fd.append('avatar_file', file);
    fd.append('platform', platform);
    showSpinner('photo-analysis-result', 'Analyzing image...');
    fetch('/api/v1/analyze/avatar', {
      method: 'POST',
      headers: { 'X-CSRFToken': csrfToken },
      body: fd
    })
      .then(r => r.json())
      .then(renderPhotoResult)
      .catch(() => {});
  }

  function analyzeAvatarUrl(url) {
    if (!url) return;
    showSpinner('photo-analysis-result', 'Fetching & analyzing image...');
    fetch('/api/v1/analyze/avatar', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
      body: JSON.stringify({ avatar_url: url, platform })
    })
      .then(r => r.json())
      .then(renderPhotoResult)
      .catch(() => {});
  }

  function renderPhotoResult(data) {
    if (!data || data.error) return;
    document.getElementById('photo-features-json').value = JSON.stringify(data);
    const sigs = [];
    if (data.photo_available !== 1 && data.photo_available !== true) {
      sigs.push({ label: 'Image could not be loaded', severity: 'warn' });
    } else {
      if (data.photo_is_default_avatar != null)
        sigs.push({ label: `Default avatar: ${data.photo_is_default_avatar ? 'Yes' : 'No'}`,
                    severity: data.photo_is_default_avatar ? 'warn' : 'ok' });
      if (data.photo_stock_score != null)
        sigs.push({ label: `Stock match: ${(data.photo_stock_score*100).toFixed(0)}%`,
                    severity: data.photo_stock_score > 0.5 ? 'bad' : data.photo_stock_score > 0.2 ? 'warn' : 'ok' });
      if (data.photo_ai_generated_score != null)
        sigs.push({ label: `AI-generated: ${(data.photo_ai_generated_score*100).toFixed(0)}%`,
                    severity: data.photo_ai_generated_score > 0.6 ? 'bad' : data.photo_ai_generated_score > 0.3 ? 'warn' : 'ok' });
      if (data.photo_width && data.photo_height)
        sigs.push({ label: `${data.photo_width}×${data.photo_height}px`, severity: 'neutral' });
      if (data.photo_color_entropy != null)
        sigs.push({ label: `Color entropy: ${data.photo_color_entropy.toFixed(2)}`, severity: 'neutral' });
      if (data.photo_has_exif != null)
        sigs.push({ label: `EXIF: ${data.photo_has_exif ? 'Present' : 'None'}`, severity: 'neutral' });
    }
    renderInlineSignals('photo-analysis-result', sigs);
  }

  function showSpinner(containerId, msg) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.style.display = '';
    el.innerHTML = `<span class="signal neutral"><span class="spinner-border spinner-border-sm me-1"></span>${msg}</span>`;
  }

  // ---------- Posts analysis ----------
  function analyzePosts(value) {
    const lines = (value || '').split('\n').map(l => l.trim()).filter(Boolean);
    if (lines.length < 2) {
      document.getElementById('posts-features-json').value = '';
      const el = document.getElementById('posts-analysis-result');
      if (el) { el.style.display = 'none'; el.innerHTML = ''; }
      return;
    }
    fetch('/api/v1/analyze/posts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
      body: JSON.stringify({ posts: value })
    })
      .then(r => r.json())
      .then(data => {
        if (data.error) return;
        document.getElementById('posts-features-json').value = JSON.stringify(data);
        const sigs = [];
        if (data.behav_post_count_analyzed != null)
          sigs.push({ label: `${data.behav_post_count_analyzed} posts analyzed`, severity: 'neutral' });
        if (data.behav_lexical_diversity != null)
          sigs.push({ label: `Lexical diversity: ${(data.behav_lexical_diversity*100).toFixed(0)}%`,
                      severity: data.behav_lexical_diversity > 0.5 ? 'ok' : 'warn' });
        if (data.behav_duplicate_post_ratio != null)
          sigs.push({ label: `Duplicates: ${(data.behav_duplicate_post_ratio*100).toFixed(0)}%`,
                      severity: data.behav_duplicate_post_ratio > 0.3 ? 'bad' : data.behav_duplicate_post_ratio > 0.05 ? 'warn' : 'ok' });
        if (data.behav_sentiment_variance != null)
          sigs.push({ label: `Sentiment var: ${data.behav_sentiment_variance.toFixed(2)}`, severity: 'neutral' });
        if (data.behav_hashtag_per_post != null)
          sigs.push({ label: `#tags/post: ${data.behav_hashtag_per_post}`,
                      severity: data.behav_hashtag_per_post > 8 ? 'warn' : 'neutral' });
        if (data.behav_url_post_ratio != null)
          sigs.push({ label: `URLs in ${(data.behav_url_post_ratio*100).toFixed(0)}% of posts`,
                      severity: data.behav_url_post_ratio > 0.7 ? 'bad' : 'neutral' });
        if (data.behav_avg_post_length != null)
          sigs.push({ label: `Avg length: ${data.behav_avg_post_length.toFixed(0)} chars`, severity: 'neutral' });
        renderInlineSignals('posts-analysis-result', sigs);
      })
      .catch(() => {});
  }

  // ---------- Derived features ----------
  function updateDerived() {
    const followers = parseFloat(document.querySelector('[name=followers]')?.value);
    const following = parseFloat(document.querySelector('[name=following]')?.value);
    const posts = parseFloat(document.querySelector('[name=posts]')?.value
                  ?? document.querySelector('[name=tweets]')?.value
                  ?? document.querySelector('[name=total_videos]')?.value);
    const ageInput = parseFloat(document.querySelector('[name=account_age_days]')?.value);
    const dateInput = document.querySelector('[name=creation_date]')?.value
                   || document.querySelector('[name=cake_day]')?.value;

    const items = [];
    if (!isNaN(followers) && !isNaN(following)) {
      const denom = following > 0 ? following : 1;
      items.push(`<span><strong>F/F ratio:</strong> ${(followers/denom).toFixed(2)}</span>`);
    }
    let ageDays = null;
    if (!isNaN(ageInput) && ageInput >= 0) ageDays = ageInput;
    else if (dateInput) {
      const d = new Date(dateInput);
      if (!isNaN(d)) ageDays = Math.round((Date.now() - d.getTime()) / 86400000);
    }
    if (ageDays != null) {
      items.push(`<span><strong>Account age:</strong> ${ageDays} days (~${(ageDays/365).toFixed(1)}y)</span>`);
      if (!isNaN(posts) && ageDays > 0) items.push(`<span><strong>Posts/day:</strong> ${(posts/ageDays).toFixed(2)}</span>`);
    }
    if (!isNaN(posts) && !isNaN(followers) && followers > 0) {
      items.push(`<span><strong>Posts/100 followers:</strong> ${(posts/followers*100).toFixed(2)}</span>`);
    }
    const panel = document.getElementById('derivedPanel');
    const content = document.getElementById('derivedContent');
    if (items.length) {
      panel.classList.remove('d-none');
      content.innerHTML = items.join('');
    } else {
      panel.classList.add('d-none');
    }
  }

  // ---------- Coverage tracker ----------
  function fieldIsFilled(el) {
    if (!el || el.disabled) return false;
    if (el.type === 'checkbox') return el.checked;
    if (el.tagName === 'SELECT' && el.multiple) {
      return Array.from(el.selectedOptions).filter(o => o.value).length > 0;
    }
    if (el.type === 'file') return el.files && el.files.length > 0;
    return (el.value || '').toString().trim() !== '';
  }

  function updateCoverage() {
    const fields = form.querySelectorAll('input, select, textarea');
    const counted = new Set();
    let total = 0;
    let filled = 0;
    fields.forEach(el => {
      const name = el.name;
      if (!name || name === 'csrf_token' || name.endsWith('_features_json')) return;
      if (counted.has(name)) return;  // count duplicates once (e.g. account_age_days)
      counted.add(name);
      total += 1;
      // For checkbox groups (cross_platforms), count as filled if ANY are checked
      if (el.type === 'checkbox' && form.querySelectorAll(`input[name="${name}"]`).length > 1) {
        if (form.querySelector(`input[name="${name}"]:checked`)) filled += 1;
      } else if (fieldIsFilled(el)) {
        filled += 1;
      }
    });
    const pct = total > 0 ? Math.round(filled / total * 100) : 0;
    document.getElementById('coverageFilled').textContent = filled;
    document.getElementById('coverageTotal').textContent = total;
    document.getElementById('coveragePct').textContent = pct;
    const bar = document.getElementById('coverageBar');
    bar.style.width = pct + '%';
    bar.classList.remove('bg-success','bg-warning','bg-danger');
    bar.classList.add(pct >= 70 ? 'bg-success' : pct >= 40 ? 'bg-warning' : 'bg-danger');

    // Confidence estimate: required fields contribute ~50%, optional adds up to +40%
    const requiredEls = form.querySelectorAll('[data-required="1"]');
    let reqFilled = 0;
    requiredEls.forEach(el => { if (fieldIsFilled(el)) reqFilled += 1; });
    const reqPct = requiredEls.length ? reqFilled / requiredEls.length : 0;
    let est = Math.round(50 + 40 * reqPct + 10 * (pct/100));
    if (est > 95) est = 95;
    document.getElementById('coverageConfidence').textContent = `~${est}%`;
    const hint = document.getElementById('coverageHint');
    if (pct < 30) hint.textContent = '— low coverage will limit accuracy';
    else if (pct < 60) hint.textContent = '— add more fields for higher accuracy';
    else hint.textContent = '— strong coverage';
  }

  // ---------- Wire up listeners ----------
  const usernameField = document.getElementById('f_username') ||
                        document.getElementById('f_name') ||
                        document.getElementById('f_channel_name');
  if (usernameField) {
    usernameField.addEventListener('blur', () => analyzeUsername(usernameField.value.trim()));
  }

  const bioField = document.getElementById('f_bio');
  if (bioField) {
    bioField.addEventListener('blur', () => analyzeBio(bioField.value));
  }

  const avatarFile = document.getElementById('f_avatar_file');
  if (avatarFile) {
    avatarFile.addEventListener('change', e => analyzeAvatarFile(e.target.files[0]));
  }
  const avatarUrl = document.getElementById('f_avatar_url');
  if (avatarUrl) {
    avatarUrl.addEventListener('blur', () => {
      const v = avatarUrl.value.trim();
      if (v) analyzeAvatarUrl(v);
    });
  }

  const postsField = document.getElementById('f_recent_posts');
  if (postsField) {
    postsField.addEventListener('blur', () => analyzePosts(postsField.value));
  }

  const derivedDeb = debounce(updateDerived, 200);
  const coverageDeb = debounce(updateCoverage, 100);
  form.addEventListener('input', () => { derivedDeb(); coverageDeb(); });
  form.addEventListener('change', () => { derivedDeb(); coverageDeb(); });
  // Initial
  updateCoverage();
  updateDerived();

  // ---------- Submit spinner ----------
  form.addEventListener('submit', () => {
    const btn = document.getElementById('submitBtn');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
    }
  });

  // ---------- Quick-fill buttons ----------
  function applyProfile(profile) {
    if (!profile) return;
    Object.entries(profile).forEach(([k, v]) => {
      // Multi-select arrays
      if (Array.isArray(v)) {
        const sel = form.querySelector(`select[name="${k}"]`);
        if (sel) {
          Array.from(sel.options).forEach(o => { o.selected = v.includes(o.value) || v.includes(o.text); });
          return;
        }
        // Cross-platform checkbox group
        if (k === 'cross_platforms') {
          form.querySelectorAll(`input[name="cross_platforms"]`).forEach(cb => {
            cb.checked = v.includes(cb.value);
          });
          return;
        }
      }
      // Booleans -> checkboxes
      if (v === true || v === false) {
        const cb = form.querySelector(`input[type="checkbox"][name="${k}"]`);
        if (cb) cb.checked = v;
        return;
      }
      // Plain inputs
      const el = form.querySelector(`[name="${k}"]`);
      if (!el) return;
      if (el.type === 'checkbox') el.checked = !!v;
      else el.value = v;
    });
    // Trigger AJAX analyses
    if (usernameField && usernameField.value) analyzeUsername(usernameField.value);
    if (bioField && bioField.value)           analyzeBio(bioField.value);
    if (postsField && postsField.value)       analyzePosts(postsField.value);
    updateCoverage(); updateDerived();
  }

  function resetForm() {
    form.reset();
    ['photo-features-json','username-features-json','bio-features-json','posts-features-json'].forEach(id => {
      const el = document.getElementById(id); if (el) el.value = '';
    });
    ['username-analysis-result','bio-analysis-result','photo-analysis-result','posts-analysis-result'].forEach(id => {
      const el = document.getElementById(id);
      if (el) { el.style.display = 'none'; el.innerHTML = ''; }
    });
    updateCoverage(); updateDerived();
  }

  document.getElementById('quickFillLegitBtn')?.addEventListener('click', () => {
    if (window.QUICK_FILL_DATA) applyProfile(window.QUICK_FILL_DATA.legit?.[platform]);
  });
  document.getElementById('quickFillFakeBtn')?.addEventListener('click', () => {
    if (window.QUICK_FILL_DATA) applyProfile(window.QUICK_FILL_DATA.fake?.[platform]);
  });
  document.getElementById('resetFormBtn')?.addEventListener('click', resetForm);

  // ---------- Live Lookup ----------
  const lookupBtn = document.getElementById('lookupBtn');
  if (lookupBtn) {
    lookupBtn.addEventListener('click', () => {
      const username = document.getElementById('lookupInput').value.trim();
      if (!username) return;
      const status = document.getElementById('lookupStatus');
      lookupBtn.disabled = true;
      lookupBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Fetching...';
      status.className = 'small mt-2 text-muted';
      status.textContent = 'Fetching profile data...';
      status.classList.remove('d-none');

      fetch(`/api/live/${platform}/${encodeURIComponent(username)}`)
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        })
        .then(data => {
          if (data.error) {
            status.className = 'small mt-2 text-danger';
            status.textContent = '✗ ' + data.error;
            return;
          }

          // Debug: log what the API returned
          console.group(`Live Lookup: ${platform}/${username}`);
          console.log('Raw response:', data);
          console.log('Features:', data.features);
          console.log('Completeness:', data.completeness);
          console.groupEnd();

          const feat = data.features || {};

          // --- Comprehensive field mapping for all platforms ---
          const map = {
            // Universal / Common fields
            username:         data.username || username,
            name:             data.name || data.username || username,
            display_name:     data.name || data.display_name || feat.display_name,
            followers:        data.followers,
            following:        data.following,
            bio:              data.bio || data.about || data.description || feat.bio || '',
            about:            data.about || data.bio || data.description || feat.about || '',
            location:         data.location || feat.location,
            website_url:      data.website || feat.website,
            avatar_url:       data.avatar_url || data.profile_pic_url || feat.avatar_url,
            account_age_days: data.account_age_days ?? feat.account_age_days,

            // Platform-specific aliases/fields
            public_repos:     data.public_repos ?? feat.public_repos,
            public_gists:     data.public_gists ?? feat.public_gists,
            company:          data.company || feat.company,
            tweets:           data.tweets ?? feat.tweets,
            posts:            data.posts ?? data.media_count ?? feat.posts,
            total_videos:     data.total_videos ?? data.videos ?? feat.total_videos,
            subscribers:      data.subscribers ?? feat.subscribers,
            channel_name:     data.channel_name || data.name,
            connections:      data.connections ?? feat.connections,
            headline:         data.headline || feat.headline,
            karma:            data.karma ?? feat.karma,
            post_karma:       data.post_karma ?? feat.post_karma,
            comment_karma:    data.comment_karma ?? feat.comment_karma,
            snap_score:       data.snap_score ?? data.score ?? feat.snap_score,
            total_stars:      data.total_stars ?? feat.star_received_total,
            total_likes:      data.total_likes ?? feat.total_likes,
            unique_languages: data.unique_languages ?? feat.unique_languages,
            contributions_year: data.contributions_year ?? feat.contributions_year,
            top_language:     data.top_language || feat.top_language,
          };

          let filled = 0;
          Object.entries(map).forEach(([fieldName, value]) => {
            if (value == null || value === '') return;
            
            // Try to find the field by name, then ID
            const el = form.querySelector(`[name="${fieldName}"]`)
                    || form.querySelector(`#f_${fieldName}`)
                    || form.querySelector(`#${fieldName}`);
            
            if (!el) return;

            if (el.type === 'checkbox') {
              el.checked = Boolean(value);
            } else if (el.tagName === 'SELECT') {
              // For connections select or other dropdowns
              const options = Array.from(el.options);
              const matchingOption = options.find(o => 
                o.value == value || o.text.toLowerCase().includes(String(value).toLowerCase())
              );
              if (matchingOption) el.value = matchingOption.value;
            } else {
              el.value = value;
            }
            filled++;
          });

          // Boolean checkboxes handling (is_verified, has_avatar)
          const booleans = [
            { key: 'is_verified', names: ['verified', 'is_verified'] },
            { key: 'has_avatar',  names: ['has_custom_avatar', 'has_avatar', 'has_photo'] }
          ];
          booleans.forEach(bool => {
            if (data[bool.key] !== undefined) {
              bool.names.forEach(n => {
                const el = form.querySelector(`[name="${n}"]`);
                if (el && el.type === 'checkbox') {
                  el.checked = Boolean(data[bool.key]);
                  filled++;
                }
              });
            }
          });

          // Update status
          status.className = 'small mt-2 text-success';
          status.textContent = `✓ Auto-filled ${filled} fields from live data`;
          if (data.completeness != null) {
            status.textContent += ` (${Math.round(data.completeness * 100)}% completeness)`;
          }

          // Show summary card if exists
          const summaryEl = document.getElementById('lookupSummary');
          if (summaryEl) {
            const items = [];
            if (data.followers != null) items.push(`${data.followers.toLocaleString()} followers`);
            if (data.posts != null) {
              const postLabel = (platform === 'twitter' || platform === 'x') ? 'tweets' : (platform === 'youtube' ? 'videos' : 'posts');
              items.push(`${data.posts.toLocaleString()} ${postLabel}`);
            }
            if (data.account_age_days) items.push(`${data.account_age_days} days old`);
            if (data.is_verified) items.push('✓ Verified');
            summaryEl.innerHTML = items.length ? `<small class="text-muted">Fetched: ${items.join(' · ')}</small>` : '';
            summaryEl.classList.remove('d-none');
          }

          // Trigger downstream analyses
          if (usernameField?.value) analyzeUsername(usernameField.value.trim());
          if (bioField?.value)      analyzeBio(bioField.value);
          const url = avatarUrl?.value?.trim();
          if (url) analyzeAvatarUrl(url);
          
          updateCoverage(); 
          updateDerived();
        })
        .catch(err => {
          status.className = 'small mt-2 text-danger';
          status.textContent = '✗ Network error: ' + err.message;
          console.error('Live lookup error:', err);
        })
        .finally(() => {
          lookupBtn.disabled = false;
          lookupBtn.innerHTML = '<i class="bi bi-cloud-download me-1"></i>Live Lookup';
        });
    });
  }
})();
