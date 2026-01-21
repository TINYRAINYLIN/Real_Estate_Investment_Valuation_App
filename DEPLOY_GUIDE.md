# 🚀 Deploy Guide - Streamlit Cloud

## ✅ Pre-Deployment Checklist

Your app is ready! Here's what you need to do:

### Step 1: Add Required Files to Git (2 min)

```bash
# Add main app files
git add app.py
git add src/pipeline/predicting_pipeline_v2.py
git add requirements.txt
git add src/exception.py
git add src/logger.py
git add src/__init__.py

# Force add artifacts (they're in .gitignore but we need them)
git add -f artifacts/feature_names.json
git add -f notebook/Best_Models/best_randomforest.pkl

# Commit
git commit -m "Add Streamlit app with Random Forest model"

# Push to GitHub
git push origin main
```

### Step 2: Deploy on Streamlit Cloud (3 min)

1. **Go to:** https://share.streamlit.io/

2. **Sign in** with your GitHub account

3. **Click "New app"** (big button in top right)

4. **Fill in the form:**
   - **Repository:** Select your repo from dropdown
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** Choose a custom name like `zillow-predictor`

5. **Click "Deploy"**

6. **Wait 2-3 minutes** while it builds

7. **Done!** Your app will be live at:
   ```
   https://YOUR-USERNAME-zillow-predictor.streamlit.app
   ```

## 🔍 Important Files Needed

Make sure these are in your GitHub repo:

### Required Files ✅
- `app.py` - Main Streamlit app
- `src/pipeline/predicting_pipeline_v2.py` - Prediction pipeline
- `src/exception.py` - Exception handling
- `src/logger.py` - Logging
- `src/__init__.py` - Package init
- `requirements.txt` - Dependencies
- `artifacts/feature_names.json` - Feature list (213 features)
- `notebook/Best_Models/best_randomforest.pkl` - Random Forest model

### Check Model File Size

```bash
# Check if model is < 100MB (GitHub limit)
dir notebook\Best_Models\best_randomforest.pkl
```

If it's > 100MB, you'll need Git LFS (see below).

## 🐛 Troubleshooting

### Issue 1: Model File Too Large (> 100MB)

**Solution: Use Git LFS**

```bash
# Install Git LFS (one time)
git lfs install

# Track .pkl files
git lfs track "*.pkl"

# Add .gitattributes
git add .gitattributes

# Add model file
git add notebook/Best_Models/best_randomforest.pkl

# Commit and push
git commit -m "Add model with Git LFS"
git push origin main
```

### Issue 2: Import Errors on Streamlit Cloud

The app already has this fix at the top:
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

### Issue 3: Missing Dependencies

Make sure `requirements.txt` has:
```
pandas
numpy
scikit-learn
streamlit
joblib
```

### Issue 4: Feature Names Not Found

Make sure you added:
```bash
git add -f artifacts/feature_names.json
```

## 📊 What Happens During Deployment

1. **Streamlit Cloud clones your repo**
2. **Installs dependencies** from requirements.txt
3. **Runs your app** with `streamlit run app.py`
4. **Assigns a public URL**
5. **Auto-deploys** on every git push

## 🎯 After Deployment

### Test Your Live App
1. Visit your app URL
2. Enter property details
3. Click "Predict Property Value"
4. Verify prediction looks reasonable

### Share Your App
- **LinkedIn:** "Check out my ML property predictor: [URL]"
- **Resume:** Add under projects with the live URL
- **Portfolio:** Embed or link to your app

### Monitor Your App
- Go to https://share.streamlit.io/
- Click on your app
- View logs, analytics, and usage

## 🔄 Update Your App

After deployment, any changes you push to GitHub will auto-deploy:

```bash
# Make changes to app.py
git add app.py
git commit -m "Update UI"
git push origin main

# Streamlit Cloud auto-deploys in ~2 minutes
```

## 💡 Pro Tips

1. **Custom Domain:** You can add a custom domain in app settings
2. **Secrets:** Use Streamlit secrets for API keys (not needed for this app)
3. **Analytics:** Check app analytics to see usage
4. **Logs:** View logs if something breaks

## ✅ Success Checklist

Before sharing your app:

- [ ] App loads without errors
- [ ] Can enter property details
- [ ] Prediction button works
- [ ] Prediction is reasonable ($50k-$2M)
- [ ] Property summary displays
- [ ] No error messages
- [ ] Works on mobile

## 🎉 You're Ready!

Run the commands in Step 1, then follow Step 2 to deploy!

**Questions?** See deploy/DEPLOY_AWS.md for the detailed AWS guide.

---

**Quick Commands:**

```bash
# Add files
git add app.py src/pipeline/predicting_pipeline_v2.py requirements.txt src/exception.py src/logger.py src/__init__.py
git add -f artifacts/feature_names.json notebook/Best_Models/best_randomforest.pkl

# Commit and push
git commit -m "Add Streamlit app with Random Forest model"
git push origin main

# Then go to: https://share.streamlit.io/
```

Good luck! 🚀
